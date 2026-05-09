"""RL configuration for the public Unitree G1 YAHMP tasks."""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from yahmp.rl import (
    ImitationLossWeights,
    ImitationTrainerCfg,
    YahmpActionMatchingPpoAlgorithmCfg,
    YahmpImitationRunnerCfg,
    YahmpKlMatchingPpoAlgorithmCfg,
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
        loss_weights=ImitationLossWeights(action=1.0, mm=0.1, reg=0.05, vq=1.0),
        trainer=ImitationTrainerCfg(lr=2e-4, grad_clip_norm=1.0),
        experiment_name="g1_yahmp_imitation",
        wandb_project="yahmp",
        wandb_tags=_wandb_tags("yahmp", "imitation", "rvq"),
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=15_000,
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
