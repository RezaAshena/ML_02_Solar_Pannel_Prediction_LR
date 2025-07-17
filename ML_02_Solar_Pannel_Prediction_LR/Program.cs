
// Create a model for predicting solar panel output using linear regression, relation between sunshine hours and revenue during the day.

//example data
var data = new[]
{
    new SunData { SunHours = 1, Revenue = 100 },
    new SunData { SunHours = 2, Revenue = 120 },
    new SunData { SunHours = 3, Revenue = 140 },
    new SunData { SunHours = 4, Revenue = 160 },
    new SunData { SunHours = 5, Revenue = 180 }
};
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