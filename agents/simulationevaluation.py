#!/usr/bin/env python3.8

import argparse
import simulation
import agent
import os
import job
import toml
import torch
import deepening
import tracing


def folder(f):
    f = os.path.join("results", f)
    if not os.path.isdir(f):
        raise argparse.ArgumentTypeError(f"{f} is not a folder")
    return f


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=folder, help="folder")
    return parser.parse_args()


def main():
    args = get_args()
    config = toml.load(os.path.join(args.folder, "config.toml"))
    sim = simulation.Simulation(config)
    agt = agent.Agent(config)
    agt.policy.load_state_dict(
        torch.load(os.path.join(args.folder, "agent.bestmodel.pth"))
    )

    print("=== Policy ===")
    print(agt.policy)

    model = job.Job(config).model
    action_set_size = len(tracing.get_all_deepen_blocks(model)) + 1

    actions = list()
    for i in range(sim.get_num_actions()):
        action, probs = agt.action(
            {
                "totaltime": sim.get_total_time(),
                "timeleft": sim.get_time_left(actions),
                "model": model,
            }
        )
        deepening.deepen_model(model, index=action)
        actions.append(action)

        print(f"=== Step {i + 1} ===")
        print(f"Action: {action}")
        print(f"Probabilities: {' '.join(f'{e:.4f}' for e in probs)}")
        print()

    actions = tuple(actions)
    no_adaptation = tuple([action_set_size - 1] * sim.get_num_actions())

    print("=== Ranking ===")
    print(
        "\n".join(
            f"{k} {v}" + (" ***" if no_adaptation == k or actions == k else "")
            for (k, v) in sim.get_acc_ranking()
        )
    )


if __name__ == "__main__":
    main()
