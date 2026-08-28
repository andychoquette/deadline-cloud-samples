# Isaac Sim / Isaac Lab SO-101 workshop image

Build recipe for the container image used by the
[`isaacsim_so101_sim_to_policy`](../../job_bundles/isaacsim_so101_sim_to_policy)
job bundle.

The image is **Isaac Lab 2.3.2 (Isaac Sim 5.1.0)** plus
[LeRobot](https://github.com/huggingface/lerobot) at the pin the
[Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
uses, plus the workshop's `sim_to_real_so101` Isaac Lab extension baked in.

## Licensing — read this first

This directory contains **only a recipe**. No NVIDIA artifact is vendored here.

- The base image `nvcr.io/nvidia/isaac-lab:2.3.2` is pulled from NVIDIA's NGC
  registry by **you**, under NVIDIA's licence terms. It is anonymously pullable —
  no NGC account or API key is required — but you must set `ACCEPT_EULA=Y` and
  `PRIVACY_CONSENT=Y` when running it (the Dockerfile bakes both in).
- The image you build contains Omniverse Kit. **Do not redistribute it.**
  Push it to a registry you control (your own Amazon ECR private repository) and
  keep it private.
- The workshop repository is Apache-2.0 and is cloned at a pinned commit during
  the build.

## Build

```bash
cd containers/isaacsim-so101-workshop
docker build -t isaacsim-so101-workshop:2.3.2 .
```

The build pulls a ~20 GB base image and finishes at roughly 50 GB. Budget disk
accordingly — a small EBS root volume is the most common reason this fails.

Override the pins if you need to:

```bash
docker build -t isaacsim-so101-workshop:2.3.2 \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2 \
  --build-arg WORKSHOP_REF=<commit-or-tag> \
  --build-arg LEROBOT_REF=<commit> \
  .
```

> Do not move the base image to Isaac Lab 3.x / Isaac Sim 6.x without checking
> your fleet first. Isaac Lab 3.x pins `isaacsim==6.0.1`, whose NVIDIA driver
> floor is above the `grid:r580` (driver 580.x) generation currently offered by
> Deadline Cloud service-managed fleets. Isaac Lab 2.3.2 declares support for
> Isaac Sim 4.5.0 / 5.0.0 / 5.1.0, which is why the bundle targets 5.1.0.

## Push to your own Amazon ECR

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-west-2
REPO=isaacsim-so101-workshop

aws ecr create-repository --repository-name "$REPO" --region "$REGION"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin \
      "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker tag isaacsim-so101-workshop:2.3.2 \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:2.3.2"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:2.3.2"
```

Then submit the job bundle with that URI:

```bash
deadline bundle submit job_bundles/isaacsim_so101_sim_to_policy \
  -p "ContainerImage=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:2.3.2" \
  -p EcrLogin=true \
  -p "OutputDir=$(pwd)/output" --yes
```

The queue's fleet role needs `ecr:GetAuthorizationToken`,
`ecr:BatchGetImage`, and `ecr:GetDownloadUrlForLayer` on that repository.

## Smoke-test locally (needs an NVIDIA GPU + the NVIDIA Container Toolkit)

```bash
docker run --rm --gpus all isaacsim-so101-workshop:2.3.2 \
  bash -lc 'python -c "import isaaclab, lerobot, sim_to_real_so101; print(\"ok\")"'
```

To list the registered tasks (should include
`Lerobot-So101-Teleop-Vials-To-Rack-DR-Eval`):

```bash
docker run --rm --gpus all isaacsim-so101-workshop:2.3.2 \
  bash -lc 'cd /workspace/Sim-to-Real-SO-101-Workshop && list_envs'
```

## What differs from the workshop's own `docker/sim/Dockerfile`

| | Workshop | This recipe |
|---|---|---|
| Workshop `source/` | bind-mounted from a developer checkout at run time | cloned into the image at a pinned commit |
| LeRobot commit | `git checkout e670ac5daf9b76` (abbreviated) | full 40-char SHA |
| Headless defaults | set per invocation | `HEADLESS=1`, `ENABLE_CAMERAS=1` baked in |
| EULA vars | passed with `-e` on every `docker run` | baked in |
| X11 / `/dev` / udev mounts | required (teleop hardware, on-screen viewport) | not used; the farm path is headless and has no leader arm |

Everything else — the `--no-deps` LeRobot install, the constraints pin, the
ffmpeg shared build, the extra X libs, the entrypoint — is upstream's, unchanged.
