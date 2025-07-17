using Microsoft.ML;
//Refactor the code to use a saved model instead of training a new one every time
//if the model exist then use that
//if not then create a new model and save it to a file for later use
//prediction of solar panel revenue based on sunshine hours

//Load the saved model
ITransformer model;//variable to storage the loaded model
var modelPath = "trainedmodel.zip";//Path the model file
var mlContext = new MLContext();//Defintition of ML Context

if (File.Exists(modelPath))
{
    using (var fileStream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
    {
        model = mlContext.Model.Load(fileStream, out var ModelInputSchema);
    }
    Console.WriteLine($"Loading model from {modelPath}");
}
else
{
    //example data
    var data = new[]
    {
    new SunData { SunHours = 1, Revenue = 100 },
    new SunData { SunHours = 2, Revenue = 120 },
    new SunData { SunHours = 3, Revenue = 140 },
    new SunData { SunHours = 4, Revenue = 160 },
    new SunData { SunHours = 5, Revenue = 180 }
};
    //Convert the data in DataView
    var trainingData = mlContext.Data.LoadFromEnumerable(data);

    //Create a regression pipeline
    var pipeline = mlContext.Transforms.Concatenate("Features", nameof(SunData.SunHours))
                       .Append(mlContext.Regression.Trainers.Sdca("Revenue"));

    //Train the data
    model = pipeline.Fit(trainingData); //longest operation we will save the model to a file for later use


    //Save the model to a file
    using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
    {
        mlContext.Model.Save(model, trainingData.Schema, fileStream);
        Console.WriteLine($"Model saved to {modelPath}");
    }

}


//Create the prediction engine
var predictionEngine = mlContext.Model.CreatePredictionEngine<SunData, RevenuePrediction>(model);

//Make a prediction
var prediction = predictionEngine.Predict(new SunData { SunHours = 6 });

Console.WriteLine($"Predicted revenue for 6 hours of sunshine: {prediction.Score}");

#region classes

public class SunData
{
    public float SunHours { get; set; } //hours of sunshine during the day
    public float Revenue { get; set; } //gains from solar panel during the day
}

public class RevenuePrediction
{
    public float Score { get; set; } //predicted revenue from solar panel
}
#endregion