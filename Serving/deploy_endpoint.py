#!/usr/bin/env python3
"""
deploy_endpoint.py — deploy Sarvam-30B on SageMaker LMI with the WORKING config.

This is the config validated 2026-09-18 (serves coherent EN+BN). It fixes the two things
that made earlier deploys fail:
  1. use the CURRENT LMI image (lmi28), NOT lmi10/0.28 (old vLLM can't run sarvam_moe →
     "container did not pass the ping health check").
  2. serve/invoke with the CHAT TEMPLATE (messages), never raw completion (which loops).

No secrets: region is a flag/env, account comes from STS, role name is a flag/env.

    pip install boto3
    AWS_PROFILE=<profile> python deploy_endpoint.py deploy   --region us-east-1
    AWS_PROFILE=<profile> python deploy_endpoint.py test     --region us-east-1
    AWS_PROFILE=<profile> python deploy_endpoint.py status   --region us-east-1
    AWS_PROFILE=<profile> python deploy_endpoint.py teardown --region us-east-1   # ALWAYS when idle
"""
import argparse, os, json, time, sys
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone

ENDPOINT = os.environ.get("SM_ENDPOINT", "sahayak-30b")
ROLE_NAME = os.environ.get("SM_ROLE", "sahayak-sagemaker-exec")
INSTANCE = os.environ.get("SM_INSTANCE", "ml.g5.12xlarge")   # bf16 30B needs 4x A10G
HF_MODEL = os.environ.get("SM_HF_MODEL", "sarvamai/Sarvam-30B")  # official Apache-2.0 base
MODEL_S3 = os.environ.get("SM_MODEL_S3", "")   # e.g. s3://sagemaker-sahayak-aps1/models/sahayak-ft-v1-ep1/ (merged weights); overrides HF_MODEL
DLC_ACCT = "763104351884"


def clients(region):
    return (boto3.client("sagemaker", region_name=region),
            boto3.client("sagemaker-runtime", region_name=region),
            boto3.client("sts", region_name=region))


def cmd_deploy(a):
    sm, _, sts = clients(a.region)
    acct = sts.get_caller_identity()["Account"]
    role = f"arn:aws:iam::{acct}:role/{ROLE_NAME}"
    image = f"{DLC_ACCT}.dkr.ecr.{a.region}.amazonaws.com/djl-inference:0.36.0-lmi28.0.0-cu130"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model, cfg = f"{ENDPOINT}-{stamp}", f"{ENDPOINT}-cfg-{stamp}"
    env = {
        **({"OPTION_MODEL_ID": MODEL_S3} if MODEL_S3 else {"HF_MODEL_ID": HF_MODEL}),   # LMI pulls s3:// prefixes directly
        "OPTION_ROLLING_BATCH": "vllm",
        "OPTION_TENSOR_PARALLEL_DEGREE": "max",
        "OPTION_TRUST_REMOTE_CODE": "true",   # sarvam_moe custom arch
        "OPTION_DTYPE": "bf16",
        "OPTION_MAX_MODEL_LEN": "4096",
        # NO OPTION_QUANTIZE — this is the bf16 base.
    }
    print(f"[image] {image}\n[model] {MODEL_S3 or HF_MODEL} -> {ENDPOINT} ({INSTANCE}, {a.region})")
    sm.create_model(ModelName=model, ExecutionRoleArn=role,
                    PrimaryContainer={"Image": image, "Environment": env})
    sm.create_endpoint_config(EndpointConfigName=cfg, ProductionVariants=[{
        "VariantName": "AllTraffic", "ModelName": model, "InstanceType": INSTANCE,
        "InitialInstanceCount": 1, "ModelDataDownloadTimeoutInSeconds": 3600,
        "ContainerStartupHealthCheckTimeoutInSeconds": 3600}])
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT)
        sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=cfg)
    except ClientError:
        sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=cfg)
    print("[endpoint] creating… (bf16 30B ≈ 60GB pull, ~15-30 min). Watch: status")


def cmd_test(a):
    _, smr, _ = clients(a.region)
    body = json.dumps({"messages": [{"role": "user",
        "content": "৪০০-এর ১৫% কত? বাংলায় ধাপে ধাপে বোঝাও।"}],
        "max_tokens": 300, "temperature": 0.0})
    r = smr.invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Body=body)
    print(r["Body"].read().decode()[:1000])
    print("\n^ must be coherent Bengali. Raw /v1/completions would loop — always use messages.")


def cmd_status(a):
    sm, _, _ = clients(a.region)
    d = sm.describe_endpoint(EndpointName=ENDPOINT)
    v = (d.get("ProductionVariants") or [{}])[0]
    print(f"{ENDPOINT} [{a.region}]: {d['EndpointStatus']} | instances {v.get('CurrentInstanceCount')} "
          f"| reason {d.get('FailureReason','-')[:120]}")


def cmd_teardown(a):
    sm, _, _ = clients(a.region)
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT); print("deleted endpoint")
    except ClientError as e:
        print("(skip endpoint)", e.response["Error"]["Code"])
    for lst, key, dele in [("list_endpoint_configs", "EndpointConfigs", "delete_endpoint_config"),
                           ("list_models", "Models", "delete_model")]:
        try:
            for it in getattr(sm, lst)(NameContains=ENDPOINT, MaxResults=50)[key]:
                name = it.get("EndpointConfigName") or it.get("ModelName")
                getattr(sm, dele)(**{list(it.keys())[0]: name})
        except Exception:
            pass
    print("torn down. Verify: status (should say Could not find endpoint).")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("deploy", cmd_deploy), ("test", cmd_test), ("status", cmd_status), ("teardown", cmd_teardown)]:
        sp = sub.add_parser(name); sp.add_argument("--region", default=os.environ.get("SM_REGION", "us-east-1"))
        sp.set_defaults(fn=fn)
    a = p.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
