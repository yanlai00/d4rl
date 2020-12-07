from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, create_sde_features_extractor, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.sac.policies import SACPolicy, Actor

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class RCRLPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(RCRLPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def _build(self, lr_schedule: Callable) -> None:
        self.actor = self.make_actor()
        actor_param = [param for name, param in self.actor.named_parameters() if "prior_network" not in name]
        self.actor.optimizer = self.optimizer_class(actor_param, lr=lr_schedule(1), **self.optimizer_kwargs)
        
        self.actor.prior_network.optimizer = self.optimizer_class(self.actor.prior_network.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return RCRLActor(**actor_kwargs).to(self.device)

class RCRLActor(Actor):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        prior_dim: int = 2, 
    ):
        super(RCRLActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor, 
            features_dim, 
            activation_fn, 
            use_sde, 
            log_std_init,
            use_expln, 
            clip_mean, 
            normalize_images,
        )
        self.prior_dim = prior_dim 
        # prior_network_layers = create_mlp(features_dim, -1, net_arch, activation_fn)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        prior_network_layers = [nn.Linear(last_layer_dim, last_layer_dim), activation_fn(), nn.Linear(last_layer_dim, self.prior_dim)]
        self.prior_network = nn.Sequential(*prior_network_layers)
        
        action_dim = get_action_dim(self.action_space)
        if self.use_sde:
            latent_sde_dim = last_layer_dim
            # Separate features extractor for gSDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                    features_dim, sde_net_arch, activation_fn
                )

            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=latent_sde_dim + self.prior_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            mu = [nn.Linear(last_layer_dim + self.prior_dim, last_layer_dim + self.prior_dim), activation_fn(), nn.Linear(last_layer_dim + self.prior_dim, action_dim)]
            self.mu = nn.Sequential(*mu)
            log_std = [nn.Linear(last_layer_dim + self.prior_dim, last_layer_dim + self.prior_dim), activation_fn(), nn.Linear(last_layer_dim + self.prior_dim, action_dim)]
            self.log_std = nn.Sequential(*log_std)

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor, return_prior=False) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation, and prior prediction and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)

        # calculate prior and combine into latent pi
        prior = self.prior_network(latent_pi)
        # self.prior_network.out = prior
        latent_pi = th.cat([latent_pi, prior], 1)
        # TODO: need to attach a loss to this

        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        if return_prior:
            return mean_actions, log_std, prior, {}
        return mean_actions, log_std, {}

    def forward_prior(self, obs: th.Tensor):
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        prior = self.prior_network(latent_pi)
        return prior

class CnnPolicy(RCRLPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )