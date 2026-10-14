# A Shiny application (bslib) predicting the arrival delay of a flight with the model of Lab 7.
# Run it from RStudio (Run App) or with: shiny::runApp("flights_app.R")
library(shiny)
library(bslib)

flights_model <- readRDS("flights_model.rds")
airlines <- flights_model$xlevels$AIRLINE

ui <- page_sidebar(
  title = "Arrival delay prediction",
  sidebar = sidebar(
    sliderInput("departure_delay", "Departure delay (min)", min = -30, max = 180, value = 0, step = 1),
    sliderInput("scheduled_time", "Scheduled flight time (min)", min = 20, max = 600, value = 120, step = 5),
    selectInput("airline", "Airline", airlines)
  ),
  value_box(title = "Predicted arrival delay", value = textOutput("prediction"), showcase = bsicons::bs_icon("airplane"))
)

server <- function(input, output) {
  output$prediction <- renderText({
    newdata <- data.frame(DEPARTURE_DELAY = input$departure_delay, SCHEDULED_TIME = input$scheduled_time,
                          AIRLINE = factor(input$airline, levels = airlines))
    sprintf("%.0f min", predict(flights_model, newdata))
  })
}

shinyApp(ui, server)
