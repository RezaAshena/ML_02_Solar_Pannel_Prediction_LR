// Create a model for predicting solar panel output using linear regression, relation between sunshine hours and revenue during the day.
using Microsoft.ML;


//example data
var data = new[]
{
    new SunData { SunHours = 1, Revenue = 100 },
    new SunData { SunHours = 2, Revenue = 120 },
    new SunData { SunHours = 3, Revenue = 140 },
    new SunData { SunHours = 4, Revenue = 160 },
    new SunData { SunHours = 5, Revenue = 180 }
};

//Defintition of ML Context
var mlContext = new MLContext();

//Convert the data in DataView
var trainingData = mlContext.Data.LoadFromEnumerable(data);

//Create a regression pipeline
var pipeline = mlContext.Transforms.Concatenate("Features", nameof(SunData.SunHours))
                   .Append(mlContext.Regression.Trainers.Sdca("Revenue"));

//Train the data
var model = pipeline.Fit(trainingData); //longest operation we will save the model to a file for later use

#region Save the model
//Save the model to a file
var modelPath = "trainedmodel.zip";
using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
{
    mlContext.Model.Save(model, trainingData.Schema, fileStream);
    Console.WriteLine($"Model saved to {modelPath}");
}
#endregion

#region reuse the  model
//Load the saved model
ITransformer loaddedModel;
using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Read, FileShare.Read))
{
  loaddedModel=mlContext.Model.Load(fileStream,out var ModelInputSchema);
}
#endregion
//Create the prediction engine
var predictionEngine = mlContext.Model.CreatePredictionEngine<SunData, RevenuePrediction>(loaddedModel);

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