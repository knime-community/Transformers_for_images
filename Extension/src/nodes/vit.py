import knime.extension as knext
import pandas as pd
from utils import knutils as kutil
from utils import modeling_utils as mutil
import logging
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    SwinForImageClassification,
    PvtForImageClassification,
    PvtImageProcessor,
    AutoImageProcessor,
)
import torch
from torch.optim import AdamW


LOGGER = logging.getLogger(__name__)


# Define sub-category
image_category = knext.category(
    path="/community/vit_ft",
    level_id="vit",
    name="Models",
    description="Vision Transformer fine-tune learner and predictor-",
    icon="icons/icon_Vision Transformer Learner.png",
)


# KNIME node definition for training
@knext.node(
    name="ViT Classification Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/icon_Vision Transformer Learner.png",
    category=image_category,
    id="img-model-learner",
)
@knext.input_table(
    name="Training Data",
    description="Table containing the training set with image column.",
)
@knext.input_table(
    name="Validation Data",
    description="Table containing the validation set with image column.",
)
@knext.output_port(
    name="Trained Model",
    port_type=kutil.classification_model_port_type,
    description="Output containing the trained Vision Transformer model.",
)
@knext.output_table(
    name="Output table",
    description="Table containing results for each epoch for train and validation.",
)
class VisionTransformerLearnerNode:
    """
    Vision Transformer Learner node

    The Vision Transformer Learner node enables users to fine-tune transformer-based models.
    It is a deep learning model that processes image data by dividing it into patches and applying transformer-based
    self-attention mechanisms.

    The model is fine-tuned using a selected image column and label column from the input training dataset. Users can
    choose from multiple Transformer architectures, including ViT, Swin Transformer, and Pyramid Transformer.
    Training parameters such as batch size, number of epochs, and learning rate can be customized. After the fine-tunig part, the node takes as input the validation set,
    which is helpful to find the best parameters of the model.
    The node outputs a fine-tuned model that can be used for image classification with the Transformer Predictor node.

    This node supports fine-tuning of three different models' architectures:

    -**Vision Transformer (ViT)**: A transformer-based model for image classification that treats images as sequences of
      patches and applies self-attention mechanisms.
      [More info](https://huggingface.co/docs/transformers/model_doc/vit)

    -**Swin Transformer**: A hierarchical transformer model with shifted window attention, designed for high-resolution
      image classification and dense prediction tasks.
      [More info](https://huggingface.co/docs/transformers/model_doc/swin)

    -**Pyramid Vision Transformer (PVT)**: A transformer model that incorporates a pyramid structure with progressively
      shrinking patch sizes, making it efficient for tasks like object detection and segmentation.
      [More info](https://huggingface.co/docs/transformers/model_doc/pvt)

    ### Configuration Options:
    -**Image Column**: Select the column containing image data (must be in PNG format).
    -**Label Column**: Select the target column containing class labels.
    -**Number of Epochs**: Defines the number of iterations over the dataset.
    -**Batch Size**: Determines the number of images processed in each training step.
    -**Learning Rate**: Sets the optimizer's step size for updating model weights.
    -**Model Choice**: Choose between ViT, Swin Transformer, or Pyramid Transformer.

    ### How It Works:
    1.The node processes images and encodes labels.
    2.The selected transformer model is initialized and fine-tuned using the provided training data.
    3.A loss function (Cross-Entropy Loss) and optimizer (AdamW) are applied to optimize model performance.
    4.Training runs for the specified number of epochs, tracking performance metrics.
    5.The trained model and a performance summary table are returned as outputs.

    """

    image_column = knext.ColumnParameter(
        label="Image Column",
        description="Select an image column for training.",
        port_index=0,
        column_filter=kutil.is_png,
    )

    label_column = knext.ColumnParameter(
        label="Label Column",
        description="Select the column containing class labels.",
        port_index=0,
        column_filter=kutil.is_numeric_or_string,
    )

    num_epochs = knext.IntParameter(
        label="Number of Epochs",
        description="Number of epochs for training the model.",
        default_value=5,
        min_value=1,
    )

    batch_size = knext.IntParameter(
        label="Batch Size",
        description="Batch size for training.",
        default_value=8,
        min_value=1,
    )

    learning_rate = knext.DoubleParameter(
        label="Learning Rate",
        description="Learning rate for the optimizer.",
        default_value=0.001,
        min_value=1e-6,
    )

    model_choice = knext.EnumParameter(
        label="Model Choice",
        description="Choose the Transformer model to use: ViT, Swin Transformer, or Pyramid Transformer.",
        default_value=mutil.ViTModelSelection.get_default().name,
        enum=mutil.ViTModelSelection,
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        training_schema: knext.Schema,
        validation_schema: knext.Schema,
    ):
        model_spec = self._create_spec(
            training_schema, validation_schema, class_probability_schema=None
        )

        # Define schema for output table
        summary_schema = knext.Schema.from_columns(
            [
                knext.Column(knext.int32(), "Epoch"),
                knext.Column(knext.double(), "Train Loss"),
                knext.Column(knext.double(), "Train Accuracy"),
                knext.Column(knext.double(), "Validation Loss"),
                knext.Column(knext.double(), "Validation Accuracy"),
            ]
        )

        return (
            model_spec,
            summary_schema,
        )  # Return the model spec and the schema for the output table

    def _create_spec(
        self,
        training_schema: knext.Schema,
        validation_schema: knext.Schema,
        class_probability_schema: knext.Schema,
    ):
        image_column = [(c.name, c.ktype) for c in training_schema if kutil.is_png(c)]

        # Check if image column have been specified
        if not image_column:
            raise knext.InvalidParametersError(
                "PNG image type column is missing in the input table."
            )

        # Preset the left most PNG image column from the input table.
        if self.image_column is None:
            self.image_column = image_column[-1][0]

        # Populate list of target columns
        label_column = [
            (c.name, c.ktype) for c in training_schema if kutil.is_nominal(c)
        ]

        # Check if the target column is available
        if not label_column:
            raise knext.InvalidParametersError("No compatible target column available.")

        # Preset the left most nominal column as target column from the input table.
        if self.label_column is None:
            self.label_column = label_column[-1][0]

        # Create schema from the target column
        target_schema = training_schema[[self.label_column]]

        # Create image column schema from the selected image column
        image_schema = training_schema[[self.image_column]]

        # Check if the option for predicting class probabilities is enabled
        if class_probability_schema is None:
            class_probability_schema = knext.Schema.from_columns("")

        # TODO: add a check that the validation schema must contain the same columns as selected for training

        return kutil.ViTClassificationModelObjectSpec(
            image_schema,
            target_schema,
            class_probability_schema,
            model_choice=self.model_choice,  # TODO make this as path to pre_trained model, either as local path or huggingface.co endpoint
        )

    def execute(
        self, exec_context: knext.ExecutionContext, training_table, validation_table
    ):
        df_train = training_table.to_pandas()
        df_val = validation_table.to_pandas()

        # Extract feature and label columns
        image_column = self.image_column
        label_column = self.label_column

        class_probability_schema = None

        # Extract images and labels from both training and validation input table.
        train_images = df_train[image_column]
        train_labels = df_train[label_column]

        val_images = df_val[image_column]
        val_labels = df_val[label_column]

        # Initialize a label encoder and encode labels
        label_enc = kutil.LabelEncoder()
        train_labels_encoded = label_enc.fit_transform(train_labels)
        val_labels_encoded = label_enc.transform(val_labels)

        # Model selection logic
        if self.model_choice == mutil.ViTModelSelection.ViT.name:
            processor = ViTImageProcessor.from_pretrained(
                mutil.ViTModelSelection.ViT.description
            )
            model = ViTForImageClassification.from_pretrained(
                mutil.ViTModelSelection.ViT.description
            )
        elif self.model_choice == mutil.ViTModelSelection.SWIN.name:
            processor = AutoImageProcessor.from_pretrained(
                mutil.ViTModelSelection.SWIN.description
            )
            model = SwinForImageClassification.from_pretrained(
                mutil.ViTModelSelection.SWIN.description
            )
        elif self.model_choice == mutil.ViTModelSelection.PYRAMID.name:
            processor = PvtImageProcessor.from_pretrained(
                mutil.ViTModelSelection.PYRAMID.description
            )
            model = PvtForImageClassification.from_pretrained(
                mutil.ViTModelSelection.PYRAMID.description
            )

        # Get class names(=column names) for probability estimates
        prob_estimates_column_names = kutil.pd.DataFrame(columns=train_labels.unique())

        # Update the table schema with probability estimates column names
        prob_estimates_column_names = prob_estimates_column_names.reindex(
            sorted(prob_estimates_column_names.columns), axis=1
        )

        input_table_schema = training_table.schema
        for column_name in prob_estimates_column_names:
            input_table_schema = input_table_schema.append(
                knext.Column(
                    ktype=knext.double(),
                    name=f"P({self.label_column}_pred={column_name})",
                )
            ).get()

        # Determine the number of classes dynamically
        num_classes = len(label_enc.classes_)

        # Create a schema that contains the class names by getting the last
        # "num_classes" number of columns from the input table.
        # class_probability_schema is then used if user wants to predict
        # probability estimates for the test data.
        if num_classes:
            class_probability_schema = input_table_schema[-num_classes:].get()

        # Create datasets and data loaders
        train_dataset = mutil.TrainingImageDataset(
            train_images, train_labels_encoded, processor
        )
        val_dataset = mutil.TrainingImageDataset(
            val_images, val_labels_encoded, processor
        )

        kutil.check_canceled(exec_context)

        train_loader = mutil.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = mutil.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        exec_context.set_progress(0.3)

        # Model setup
        if self.model_choice == mutil.ViTModelSelection.ViT.name:
            backbone = model.vit
            model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
        elif self.model_choice == mutil.ViTModelSelection.SWIN.name:
            backbone = model.swin
            model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
        elif self.model_choice == mutil.ViTModelSelection.PYRAMID.name:
            backbone = model.pvt
            model.classifier = torch.nn.Linear(
                model.config.hidden_sizes[-1], num_classes
            )

        for p in backbone.parameters():
            p.requires_grad = False

        for p in model.classifier.parameters():
            p.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate
        )

        # Training loop with table creation
        training_summary = []

        kutil.check_canceled(exec_context)

        for epoch in range(self.num_epochs):
            model.train()
            total_loss_train, total_acc_train = 0.0, 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(pixel_values=images).logits
                labels = labels.to(mutil.device)
                loss = criterion(outputs, labels.long())
                acc = (outputs.argmax(dim=1) == labels).sum().item()
                loss.backward()
                optimizer.step()
                total_loss_train += loss.item()
                total_acc_train += acc

            kutil.check_canceled(exec_context)
            avg_train_loss = total_loss_train / len(train_loader)
            avg_train_acc = total_acc_train / len(df_train)

            model.eval()
            total_loss_val, total_acc_val = 0.0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(pixel_values=images).logits
                    labels = labels.to(
                        mutil.device
                    )  # Convert labels to the correct device
                    loss = criterion(outputs, labels.long())
                    acc = (outputs.argmax(dim=1) == labels).sum().item()
                    total_loss_val += loss.item()
                    total_acc_val += acc

            avg_val_loss = total_loss_val / len(val_loader)
            avg_val_acc = total_acc_val / len(df_val)

            training_summary.append(
                [epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]
            )

        summary_df = pd.DataFrame(
            training_summary,
            columns=[
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Validation Loss",
                "Validation Accuracy",
            ],
        )

        # Force column types to match `configure()`
        summary_df = summary_df.astype(
            {
                "Epoch": "int32",
                "Train Loss": "float64",
                "Train Accuracy": "float64",
                "Validation Loss": "float64",
                "Validation Accuracy": "float64",
            }
        )

        trained_model = kutil.ViTClassificationModelObject(
            spec=self._create_spec(
                input_table_schema, input_table_schema, class_probability_schema
            ),
            model=model,
            label_enc=label_enc,
        )

        return trained_model, knext.Table.from_pandas(summary_df)


# General settings for classification predictor node
@knext.parameter_group(label="Output")
class ClassificationPredictorGeneralSettings:
    prediction_column = knext.StringParameter(
        "Custom prediction column name",
        "If no name is specified for the prediction column, it will default to `<target_column_name>_pred`.",
        default_value="",
    )

    predict_probs = knext.BoolParameter(
        "Predict probability estimates",
        "Predict probability estimates for each target class.",
        True,
    )

    prob_columns_suffix = knext.StringParameter(
        "Suffix for probability columns",
        "Allows to add a suffix for the class probability columns.",
        default_value="",
    )


# KNIME node definition for prediction
@knext.node(
    name="ViT Classification Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/icon_Vision Transformer Predictor.png",
    category=image_category,
    id="img-model-predictor",
)
@knext.input_port(
    name="Trained Model",
    port_type=kutil.classification_model_port_type,
    description="Input containing the trained Vision Transformer model.",
)
@knext.input_table(
    name="Input Data",
    description="Table containing an image column.",
)
@knext.output_table(
    name="Output Data",
    description="The input table concatenated with a predictions column and optionally class probabilities.",
)
class VisionTransformerPredictor:
    """
    Vision Transformer Predictor

    The Vision Transformer Predictor node applies a fine-tuned Transformer model to classify images in the given input dataset.
    It computes the predicted class for each image and, optionally, provides class probability estimates.

    The node requires a fine-tuned Transformer model from the Vision Transformer Learner node and a dataset containing image data.
    The prediction column name can be customized, and the node supports multiple Transformer architectures.

    It is only executable if the test data contains the image column that was used by the learner model.

    ### Configuration Options:

    - **Custom Prediction Column Name**: Specify a custom name for the prediction column.
    - **Predict Probability Estimates**: Enable to obtain confidence scores for each class.
    - **Probability Columns Suffix**: Add a suffix to probability columns for easy identification.

    ### How It Works:

    1. The node extracts images from the test dataset.
    2. The selected transformer model processes the images and makes predictions.
    3. The highest probability class is assigned as the predicted label.
    4. (Optional) Class probability estimates are computed using softmax and included in the output.

    """

    predictor_settings = ClassificationPredictorGeneralSettings()

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        model_spec: kutil.ViTClassificationModelObjectSpec,
        table_schema: knext.Schema,
    ):
        y_pred = kutil.get_prediction_column_name(
            self.predictor_settings.prediction_column, model_spec.target_schema
        )

        # Add prediction column names in the schema
        for column_name in y_pred:
            table_schema = table_schema.append(
                knext.Column(ktype=knext.string(), name=column_name)
            )

        # Add probability estimate column names in the schema
        if self.predictor_settings.predict_probs:
            for column in model_spec.class_probability_schema:
                if self.predictor_settings.prob_columns_suffix:
                    table_schema = table_schema.append(
                        knext.Column(
                            ktype=column.ktype,
                            name=f"{column.name}{self.predictor_settings.prob_columns_suffix}",
                        )
                    )
                else:
                    table_schema = table_schema.append(
                        knext.Column(ktype=column.ktype, name=column.name)
                    )

        return table_schema

    def execute(
        self,
        exec_context: knext.ExecutionContext,
        model_port: kutil.ViTClassificationModelObject,
        input_table,
    ):
        df = input_table.to_pandas()

        # Get list target column for the prediction column
        y_pred = kutil.get_prediction_column_name(
            self.predictor_settings.prediction_column, model_port.spec.target_schema
        )

        image_col_names_list = (
            model_port.spec.image_schema.column_names
        )  # list containing exactly one column name

        images = df[image_col_names_list[0]]

        model_name = model_port.spec.model_choice

        if model_name == mutil.ViTModelSelection.ViT.name:
            processor = ViTImageProcessor.from_pretrained(
                mutil.ViTModelSelection.ViT.description
            )
        elif model_name == mutil.ViTModelSelection.SWIN.name:
            processor = AutoImageProcessor.from_pretrained(
                mutil.ViTModelSelection.SWIN.description
            )
        elif model_name == mutil.ViTModelSelection.PYRAMID.name:
            processor = PvtImageProcessor.from_pretrained(
                mutil.ViTModelSelection.PYRAMID.description
            )

        dataset = mutil.PredictionImageDataset(images, processor)
        dataloader = mutil.DataLoader(dataset, batch_size=8, shuffle=False)

        model_port._model.eval()
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(model_port._model.device)
                outputs = model_port._model(pixel_values=batch).logits
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                all_predictions.extend(predictions)

                if self.predictor_settings.predict_probs:
                    probabilities = (
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )
                    all_probabilities.extend(probabilities)

        decoded_predictions = model_port.decode_target_values(
            kutil.pd.Series(all_predictions)
        )

        decoded_predictions.index = df.index
        df[y_pred[0]] = decoded_predictions

        if self.predictor_settings.predict_probs:
            if self.predictor_settings.prediction_column:
                # Get original target column name (and not the custom name)
                # for class probability column names
                y_pred = kutil.get_prediction_column_name(
                    "", model_port.spec.target_schema
                )

            class_probability_names = model_port.get_class_probability_column_names(
                y_pred, self.predictor_settings.prob_columns_suffix
            )
            for col, probs in zip(class_probability_names, zip(*all_probabilities)):
                df[col] = probs

        return knext.Table.from_pandas(df)
