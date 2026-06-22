"""RL configuration for the public Unitree G1 YAHMP tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from yahmp.rl import (
    ImitationLossWeights,
    ImitationTrainerCfg,
    YahmpActionMatchingPpoAlgorithmCfg,
    YahmpImitationRunnerCfg,
    YahmpKlMatchingPpoAlgorithmCfg,
    YahmpLocomotionOnPolicyRunnerCfg,
    YahmpOnPolicyRunnerCfg,
    YahmpStudentOnPolicyRunnerCfg,
)


def _wandb_tags(*extra: str) -> tuple[str, ...]:
    return extra


def unitree_g1_yahmp_teacher_ppo_runner_cfg() -> YahmpOnPolicyRunnerCfg:
    return YahmpOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpFutureActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpFutureCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_yahmp_teacher",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "teacher", "privileged"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_ppo_runner_cfg() -> YahmpOnPolicyRunnerCfg:
    return YahmpOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_yahmp",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "history_encoder", "residual_actions"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_no_res_ppo_runner_cfg() -> YahmpOnPolicyRunnerCfg:
    return YahmpOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_yahmp_no_res",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "history_encoder", "no_residual"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_encdec_ppo_runner_cfg() -> YahmpOnPolicyRunnerCfg:
    return YahmpOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpEncoderDecoderActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpEncoderDecoderCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_yahmp_encdec",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags(
            "yahmp", "history_encoder", "encoder_decoder", "state_injection"
        ),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_future_ppo_runner_cfg() -> YahmpOnPolicyRunnerCfg:
    return YahmpOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpFutureActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpFutureCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_yahmp_future",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags(
            "yahmp", "future_encoder", "history_encoder", "residual_actions"
        ),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_student_action_matching_rl_runner_cfg() -> (
    YahmpStudentOnPolicyRunnerCfg
):
    return YahmpStudentOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.student_teacher_policy:YahmpStudentTeacherActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.4,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=YahmpActionMatchingPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0025,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.005,
            max_grad_norm=1.0,
            bc_coef_start=1.0,
            bc_coef_end=0.05,
            bc_anneal_iters=20_000,
            bc_loss_type="mse",
        ),
        experiment_name="g1_yahmp_student_action_matching_rl",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "student", "action_matching_rl"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=20_000,
        obs_groups={
            "actor": ("actor",),
            "critic": ("critic",),
            "teacher": ("teacher_actor",),
        },
    )


def unitree_g1_yahmp_imitation_runner_cfg() -> YahmpImitationRunnerCfg:
    return YahmpImitationRunnerCfg(
        seed=1,
        expert=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpEncoderDecoderActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        student=RslRlModelCfg(
            class_name="yahmp.rl.imitation_RVQ_policy:YahmpImitationModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        loss_weights=ImitationLossWeights(action=10.0, mm=1.0, reg=0.05, vq=1.0),
        trainer=ImitationTrainerCfg(
            lr=2e-4,
            grad_clip_norm=1.0,
            mm_warmup_steps=10_000,
            mm_start=0.1,
            mm_end=1.0,
        ),
        experiment_name="g1_yahmp_imitation",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "imitation", "rvq"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=15_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_imitation_residual_runner_cfg() -> YahmpImitationRunnerCfg:
    cfg = unitree_g1_yahmp_imitation_runner_cfg()
    cfg.trainer = ImitationTrainerCfg(lr=2e-4, grad_clip_norm=1.0)
    cfg.experiment_name = "g1_yahmp_imitation_residual"
    cfg.wandb_tags = _wandb_tags("yahmp", "imitation", "rvq", "residual")
    return cfg


def unitree_g1_yahmp_locomotion_runner_cfg() -> YahmpLocomotionOnPolicyRunnerCfg:
    return YahmpLocomotionOnPolicyRunnerCfg(
        seed=1,
        # Number of RVQ codebooks the categorical drives (RVQ stays 8-deep for
        # checkpoint compat). None => all 8 (baseline). Override from the CLI,
        # e.g. `--agent.rvq-num-active-quantizers 3`, to test a coarser subset.
        rvq_num_active_quantizers=None,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.locomotion_policy:YahmpLocomotionActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=None,
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name="yahmp.rl.categorical_ppo:YahmpCategoricalPPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # NOTE: categorical entropy is ~55 nat at uniform (8 heads × log(1024)).
            # With coef=0.01 the entropy bonus dominated the surrogate loss ~9x,
            # pinning the policy near uniform. 0.002 gives a bonus ~2x the
            # surrogate magnitude → enough exploration pressure without freeze.
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=8,  # was 4 — categorical params are 64x bigger than Gaussian
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_yahmp_locomotion",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "locomotion", "frozen_imitation", "high_level"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=15_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_locomanip_runner_cfg() -> YahmpLocomotionOnPolicyRunnerCfg:
    """Runner for the point-goal hand-reach (loco-manipulation) task.

    Identical hierarchical setup to the locomotion runner (frozen imitation
    backbone + categorical high-level + critic); only the experiment name and
    tags differ. The actor obs/goal dims (the reach command is 5-D vs the 3-D
    twist) are derived automatically by the runner from the env.
    """
    cfg = unitree_g1_yahmp_locomotion_runner_cfg()
    cfg.experiment_name = "g1_yahmp_locomanip"
    cfg.wandb_tags = _wandb_tags(
        "yahmp", "locomanip", "reach", "frozen_imitation", "high_level"
    )
    return cfg


def unitree_g1_yahmp_boxing_runner_cfg() -> YahmpLocomotionOnPolicyRunnerCfg:
    """Runner for the boxing task (velocity tracking + commanded-hand striking).

    Identical hierarchical setup to the locomotion runner (frozen imitation
    backbone + categorical high-level + critic); only the experiment name and
    tags differ. The actor obs/goal dims (the boxing command is 8-D: velocity
    3-D + strike 5-D) are derived automatically by the runner from the env.
    """
    cfg = unitree_g1_yahmp_locomotion_runner_cfg()
    cfg.experiment_name = "g1_yahmp_boxing"
    cfg.wandb_tags = _wandb_tags(
        "yahmp", "boxing", "strike", "frozen_imitation", "high_level"
    )
    return cfg


def unitree_g1_yahmp_navigation_runner_cfg() -> YahmpLocomotionOnPolicyRunnerCfg:
    """Runner for the continuous point-to-point navigation task.

    Identical hierarchical setup to the locomotion runner (frozen imitation
    backbone + categorical high-level + critic); only the experiment name and
    tags differ. The actor obs/goal dims (the navigation command is 3-D:
    heading-frame goal x_b, y_b, dist) are derived automatically by the runner
    from the env.
    """
    cfg = unitree_g1_yahmp_locomotion_runner_cfg()
    cfg.experiment_name = "g1_yahmp_navigation"
    cfg.wandb_tags = _wandb_tags(
        "yahmp", "navigation", "point_to_point", "frozen_imitation", "high_level"
    )
    return cfg


def unitree_g1_yahmp_balance_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Runner for the balance (stand-still) task.

    A PLAIN continuous Gaussian-MLP policy with standard PPO -- deliberately NOT
    the YAHMP categorical/RVQ stack (no imitation checkpoint). The continuous
    deterministic-mean inference is what makes the stand jitter-free.
    """
    return RslRlOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_yahmp_balance",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "balance", "stand", "continuous"),
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=5_000,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
    )


def unitree_g1_yahmp_student_kl_matching_rl_runner_cfg() -> (
    YahmpStudentOnPolicyRunnerCfg
):
    return YahmpStudentOnPolicyRunnerCfg(
        seed=1,
        actor=RslRlModelCfg(
            class_name="yahmp.rl.student_teacher_policy:YahmpStudentTeacherActorModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.4,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            class_name="yahmp.rl.policy:YahmpCriticModel",
            hidden_dims=(512, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=YahmpKlMatchingPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
            kl_coef=0.1,
            kl_coef_min=0.01,
            kl_coef_anneal_iters=60_000,
        ),
        experiment_name="g1_yahmp_student_kl_matching_rl",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "student", "kl_matching_rl"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=20_000,
        obs_groups={
            "actor": ("actor",),
            "critic": ("critic",),
            "teacher": ("teacher_actor",),
        },
    )
