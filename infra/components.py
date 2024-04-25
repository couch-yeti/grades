from typing import Optional
import random
from uuid import uuid4

from aws_cdk import CfnOutput
from aws_cdk.aws_sagemaker_alpha import (
    ContainerImage,
    ContainerDefinition,
    Endpoint,
    EndpointConfig,
    Model,
    ModelData,
)
from constructs import Construct
from pydantic import Field, BaseModel


HF_IMAGE_TEMPLATE = "{registry}.dkr.{hostname}/{repository}"


def generate_unique_id(chars=10):
    characters = str(uuid4())
    return "".join(random.choice(characters) for _ in range(chars))


class HFVars(BaseModel):
    """Vars can be read about here:
    https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/launcher
    """

    HF_MODEL_ID: str


class Props(BaseModel):
    """The available TGI Images can be found here:
    https://github.com/aws/deep-learning-containers/releases?q=tgi+AND+gpu&expanded=true
    """

    instance_type: str = Field(
        description="The ec2 instance type to be used for the sagemaker endpoint",
        default="ml.g5.2xlarge",
    )
    environment_vars: HFVars = Field(
        description="The acceptable arguments for the Huggingface TGI container"
    )
    s3_model_path: Optional[str] = Field(
        description="Parameter to use if loading a fine tuned model from S3",
        default=None,
    )
    tag: str = Field(
        description="The image tag to be used (see available tgi images link)"
    )
    start_up_health_check_seconds: int = Field(
        description="How long to wait for the health check to succeed (crucial for big models)",
        default=300,
    )
    repo_name: str = Field(
        description="The name of the ECS respository where the image can be pulled",
        default="huggingface-pytorch-inference",
    )


class TgiLlm(Construct):
    def __init__(self, scope, cid, props: Props):
        super().__init__(scope=scope, id=cid)
        self.props = props
        self.unique_id = generate_unique_id()
        self.model_name = f"{self.props.name}-{self.unique_id}"
        self.endpoint_name = f"{self.props.name}-endpoint-{self.unique_id}"
        self.endpoint_config_name = (
            f"{self.props.name}-config-{self.unique_id}"
        )

    def run_endpoint_build(self):

        self._set_container()
        self._set_model()
        self._set_config()
        self._set_endpoint()

        CfnOutput(self, "EndpointName", value=self.endpoint.endpoint_name)

    def _set_container(self):
        container_image = ContainerImage.from_dlc(
            self.props.repo_name, tag=self.props.tag
        )
        self.container_definition = ContainerDefinition(
            image=container_image, environment=self.props.environment_vars
        )
        if self.props.s3_model_path:
            model_data = ModelData.from_asset(self.props.s3_model_path)
            self.container_definition.model_data = model_data

    def _set_model(self):
        self.model = Model(
            self,
            "TgiModel",
            model_name=self.model_name,
            containers=[self.container_definition],
        )

    def _set_config(self):
        self.endpoint_config = EndpointConfig(
            self,
            "TgiEndpointConfig",
            endpoint_config_name=self.endpoint_config_name,
            instance_production_variants=[
                {
                    "modelName": self.model.model_name,
                    "variantName": "primary",
                    "initialVariantWeight": "1.0",
                    "initialInstanceCount": 1,
                    "instanceType": self.props.instance_type,
                    "containerStartupHealthCheckTimeoutInSeconds": self.props.start_up_health_check_seconds,
                }
            ],
        )

    def _set_endpoint(self):
        self.endpoint = Endpoint(
            self,
            "TgiEndpoint",
            endpoint_name=self.endpoint_name,
            endpoint_config=self.endpoint_config,
        )
