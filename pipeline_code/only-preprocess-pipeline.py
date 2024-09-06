from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterString
import sagemaker

# Define SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Define S3 bucket and directories
s3_bucket = "sagemaker_bucket"
raw_data = f"s3://{s3_bucket}/raw_data/rental_1000.csv"
processed_data = f"s3://{s3_bucket}/processed_data/"
script_path = f"s3://{s3_bucket}/scripts/preprocess.py"  # New script path in S3lfc

# Define pipeline parameters
input_data = ParameterString(name="InputData", default_value=raw_data)

# Preprocessing Step - Using preprocess.py stored in S3
sklearn_processor = SKLearnProcessor(framework_version='0.23-1',
                                     instance_type='ml.m5.xlarge',
                                     instance_count=1,
                                     base_job_name='rental-preprocessing',
                                     role=role)

step_preprocess = ProcessingStep(
    name="PreprocessingStep",
    processor=sklearn_processor,
    inputs=[sagemaker.processing.ProcessingInput(source=input_data,
                                                 destination='/opt/ml/processing/input')],
    outputs=[sagemaker.processing.ProcessingOutput(output_name="processed_data",
                                                   source='/opt/ml/processing/output',
                                                   destination=processed_data)],
    code=script_path  # Updated to use the S3 path of the script
)

# Create a pipeline with only the preprocessing step
pipeline = Pipeline(
    name="RentalPredictionPreprocessingPipeline",
    steps=[step_preprocess]
)

# Upsert the pipeline
pipeline.upsert(role_arn=role)

print(f"Pipeline {pipeline.name} created/updated successfully.")
