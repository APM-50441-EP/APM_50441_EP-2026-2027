# A REST API serving the arrival delay model of Lab 7 (plumber2).
# Run it with: plumber2::api("flights_api.R") |> plumber2::api_run(port = 8000)
# and query it with: http://127.0.0.1:8000/predict?departure_delay=30&scheduled_time=180&airline=AA

flights_model <- readRDS("flights_model.rds")

#* Predict the arrival delay of a flight
#* @get /predict
#* @query departure_delay:number Departure delay (minutes)
#* @query scheduled_time:number Scheduled flight time (minutes)
#* @query airline:string Airline IATA code (AA, DL, WN...)
#* @serializer unboxedJSON
function(query) {
  if (is.null(query$airline) || !query$airline %in% flights_model$xlevels$AIRLINE) {
    return(list(error = "unknown airline"))
  }
  newdata <- data.frame(DEPARTURE_DELAY = query$departure_delay, SCHEDULED_TIME = query$scheduled_time,
                        AIRLINE = factor(query$airline, levels = flights_model$xlevels$AIRLINE))
  list(arrival_delay = unname(predict(flights_model, newdata)))
}
