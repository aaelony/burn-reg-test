#![recursion_limit = "256"]

// Logistic Regression using the Burn framework and a Huggingface Candle backend.

use burn::{
    backend::{
        Autodiff,
        candle::{Candle, CandleDevice},
    },
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::sigmoid, backend::Backend},
};

type DefaultDevice = CandleDevice;
type DefaultBackend = Candle;
type MyBackend = Autodiff<DefaultBackend>;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Module, Debug)]
pub struct LogisticRegression<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LogisticRegression<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear = LinearConfig::new(2, 1).with_bias(true).init(device);
        Self { linear }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        sigmoid(self.linear.forward(input))
    }
}

fn main() {
    // Initialize the device
    //  let device = MyBackend::default();
    let device = DefaultDevice::Cpu;

    // Create training data for XOR problem
    let x_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_data = [[0.0], [1.0], [1.0], [0.0]];

    // Convert to tensors - using concrete types
    let x_tensor = Tensor::<MyBackend, 2>::from_floats(x_data, &device);
    let y_tensor = Tensor::<MyBackend, 2>::from_floats(y_data, &device);

    println!("Training data:");
    println!("X shape: {:?}", &x_tensor.shape());
    println!("Y shape: {:?}", &y_tensor.shape());

    // Initialize model
    let model = LogisticRegression::<MyBackend>::new(&device);

    println!("\nComputing initial predictions...");

    // Forward pass
    let predictions = model.forward(x_tensor.clone());

    // Calculate loss
    let loss = binary_cross_entropy_loss(predictions.clone(), y_tensor.clone());
    let loss_value = loss.into_scalar();
    println!("Initial Loss: {:.6}", loss_value);

    // Test the model
    println!("\nTesting the model:");
    let test_predictions = model.forward(x_tensor.clone());

    println!("Input -> Predicted -> Actual");
    for i in 0..4 {
        let input = x_tensor.clone().slice([i..i + 1, 0..2]);
        let pred = test_predictions.clone().slice([i..i + 1, 0..1]);
        let actual = y_tensor.clone().slice([i..i + 1, 0..1]);

        let input_data = input.to_data();
        let pred_data = pred.to_data();
        let actual_data = actual.to_data();

        let input_vals = input_data.to_vec::<f32>().unwrap();
        let pred_val = pred_data.to_vec::<f32>().unwrap()[0];
        let actual_val = actual_data.to_vec::<f32>().unwrap()[0];

        println!(
            "[{:.1}, {:.1}] -> {:.4} -> {:.1}",
            input_vals[0], input_vals[1], pred_val, actual_val
        );
    }
}

fn binary_cross_entropy_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    // p = predictions
    // -(y * log(p) + (1-y) * log(1-p)
    let epsilon = 1e-7;
    let predictions = predictions.clamp(epsilon, 1.0 - epsilon);
    let positive_part = targets.clone() * predictions.clone().log();
    let negative_part =
        (targets.ones_like() - targets) * (predictions.ones_like() - predictions).log();
    let loss = -(positive_part + negative_part);
    loss.mean()
}
