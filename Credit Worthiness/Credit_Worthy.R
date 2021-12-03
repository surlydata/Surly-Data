
# Refs ----
# pre-processing recommendations
#   https://www.tmwr.org/pre-proc-table.html


# Libraries ----
library(tidyverse)
library(tidymodels)
library(glmnet)
library(randomForest)
library(kernlab)
library(vip)
library(readxl)
library(writexl)
library(skimr)
library(corrr)

tidymodels_prefer()

# Parallel processing ----
# not implemented


# Import ----

credit_data_training <- read_excel("../credit-data-training.xlsx")
customers_to_score <- read_excel("../customers-to-score.xlsx")


# Inspect and clean ----

# credit_data_training - check and fix 
glimpse(credit_data_training)
skim(credit_data_training)

credit_data_training <- credit_data_training %>% 
    rename_with( ., ~ str_replace_all(., "-", "_") )


# customers_to_score - check and fix
glimpse(customers_to_score)
skim(customers_to_score)

customers_to_score <- customers_to_score %>% 
    rename_with( ., ~ str_replace_all(., "-", "_") )


# basic pre-processing for classification
credit_data_training <- credit_data_training %>% 
    mutate_if(is_character, as_factor)

skim(credit_data_training)

cor_table <- credit_data_training %>% 
    select( where(is.numeric) ) %>% 
    correlate()
cor_table

cor_table2 <- cor_table %>%
    pivot_longer( -term, names_to = "colname", values_to = "cor" ) %>%
    mutate(cor_abs = abs(cor)) %>% 
    arrange( desc(cor_abs) ) %>%
    filter( row_number() %% 2 == 1 )
cor_table2


# Spend data ----

table(credit_data_training$Credit_Application_Result)

set.seed(123)
credit_split <- initial_split(credit_data_training,
                              prop = 0.70,
                              strata = Credit_Application_Result)


# Functions ----

print_roc_curve <- function( roc_tibble ){
    roc_tibble %>% 
        ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
        geom_path(size = 1) +
        geom_abline(lty = 3) + 
        coord_equal() +
        theme_classic()
}


# Models ----


## Cross Validation ----

# same folds for all 
set.seed(123)
models_folds <- vfold_cv(training(credit_split), v = 10)


## Train ----

### logistic regression ----

# define model, recipe, workflow and tuning

logistic_model <- 
    logistic_reg(
        penalty = tune(),
        mixture = tune()) %>%
    set_engine("glmnet") %>% 
    set_mode("classification")

logistic_grid <- grid_regular(
    parameters(logistic_model), levels = 5)

logistic_recipe <- 
    training(credit_split) %>% 
    recipe(Credit_Application_Result ~ .) %>% 
    step_impute_median(Age_years) %>% 
    step_rm(Duration_in_Current_address) %>% 
    step_zv(all_predictors()) %>%
    step_dummy(all_nominal_predictors())

logistic_wflow <- 
    workflow() %>%
    add_model(logistic_model) %>% 
    add_recipe(logistic_recipe)

set.seed(123)
logistic_tuned <- 
    logistic_wflow %>% 
    tune_grid(
        grid = logistic_grid,
        resamples = models_folds,
        control = control_grid(save_pred = TRUE),
        metrics = metric_set(roc_auc, accuracy)
    )

# check the tuning
logistic_tuned %>% 
    collect_metrics()
logistic_tuned %>% 
    collect_predictions()
autoplot(logistic_tuned)

# select best model
logistic_tuned_best <- logistic_tuned %>% 
    select_best("roc_auc")
logistic_tuned_best

# prep data to get the AUC and accuracy metrics
logistic_train_preds <- 
    logistic_tuned %>% 
    collect_predictions(parameters = logistic_tuned_best)

# create tibble and add row to model metrics
logistic_train_metrics_tbl <- tibble(
    model = "Logistic Regression",
    sample = "CV",
    roc_auc = logistic_train_preds %>%
        roc_auc(Credit_Application_Result, .pred_Creditworthy) %>%
        pull(.estimate) %>% round( . , 3),
    accuracy = logistic_train_preds %>%
        accuracy(Credit_Application_Result, .pred_class) %>%
        pull(.estimate) %>% round( . , 3)
)

# create a tibble for plotting
logistic_train_tbl <- 
    logistic_train_preds %>% 
    roc_curve(Credit_Application_Result, .pred_Creditworthy) %>% 
    mutate(model = str_c(
        "Logistic Reg - train - ",
        "AUC: ", logistic_train_metrics_tbl$roc_auc,
        " - ",
        "Accuracy: ", logistic_train_metrics_tbl$accuracy))

# plot
print_roc_curve( logistic_train_tbl )


### random forest ----

# define model, recipe, workflow and tuning

rand_forest_model <- 
    rand_forest(
        mtry = sqrt(.cols()),
        trees = tune(),
        min_n = tune()) %>% 
    set_engine("randomForest") %>% 
    set_mode("classification")

rand_forest_grid <- grid_regular(
    parameters(rand_forest_model), levels = 5)

rand_forest_recipe <- 
    training(credit_split) %>% 
    recipe(Credit_Application_Result ~ .) %>% 
    step_impute_median(Age_years) %>% 
    step_rm(Duration_in_Current_address) %>% 
    step_zv(all_predictors())

rand_forest_wflow <- 
    workflow() %>%
    add_model(rand_forest_model) %>% 
    add_recipe(rand_forest_recipe)

set.seed(123)
rand_forest_tuned <- 
    rand_forest_wflow %>% 
    tune_grid(
        grid = rand_forest_grid,
        resamples = models_folds,
        control = control_grid(save_pred = TRUE),
        metrics = metric_set(roc_auc, accuracy)
    )

# check the tuning
rand_forest_tuned %>% 
    collect_metrics()
rand_forest_tuned %>% 
    collect_predictions()
autoplot(rand_forest_tuned)

# select best model
rand_forest_tuned_best <- rand_forest_tuned %>% 
    select_best("roc_auc")
rand_forest_tuned_best

# prep data to get the AUC and accuracy metrics
rand_forest_train_preds <- 
    rand_forest_tuned %>% 
    collect_predictions(parameters = rand_forest_tuned_best)

# create tibble and add row to model metrics
rand_forest_train_metrics_tbl <- tibble(
    model = "Random Forest",
    sample = "CV",
    roc_auc = rand_forest_train_preds %>%
        roc_auc(Credit_Application_Result, .pred_Creditworthy) %>%
        pull(.estimate) %>% round( . , 3),
    accuracy = rand_forest_train_preds %>%
        accuracy(Credit_Application_Result, .pred_class) %>%
        pull(.estimate) %>% round( . , 3)
)

# create a tibble for plotting
rand_forest_train_tbl <- 
    rand_forest_train_preds %>% 
    roc_curve(Credit_Application_Result, .pred_Creditworthy) %>% 
    mutate(model = str_c(
        "Random Forest - train - ",
        "AUC: ", rand_forest_train_metrics_tbl$roc_auc,
        " - ",
        "Accuracy: ", rand_forest_train_metrics_tbl$accuracy))

# plot
print_roc_curve( rand_forest_train_tbl )


### support vm ----


# define model, recipe, workflow and tuning

svm_rbf_model <- 
    svm_rbf(
        cost = tune(),
        rbf_sigma = tune(),
        margin = tune()) %>% 
    set_engine("kernlab") %>% 
    set_mode("classification")

svm_rbf_grid <- grid_regular(
    parameters(svm_rbf_model), levels = 3)

svm_rbf_recipe <- 
    training(credit_split) %>% 
    recipe(Credit_Application_Result ~ .) %>% 
    step_impute_median(Age_years) %>% 
    #step_rm(Duration_in_Current_address, Concurrent_Credits) %>% 
    step_rm(Duration_in_Current_address) %>% 
    step_zv(all_predictors()) %>% 
    step_dummy(all_nominal_predictors()) %>% 
    #step_zv(all_nominal_predictors()) %>% 
    step_normalize(all_predictors())

svm_rbf_wflow <- 
    workflow() %>%
    add_model(svm_rbf_model) %>% 
    add_recipe(svm_rbf_recipe)

set.seed(123)
svm_rbf_tuned <- 
    svm_rbf_wflow %>% 
    tune_grid(
        grid = svm_rbf_grid,
        resamples = models_folds,
        control = control_grid(save_pred = TRUE),
        metrics = metric_set(roc_auc, accuracy)
    )

# check the tuning
svm_rbf_tuned %>% 
    collect_metrics()
svm_rbf_tuned %>% 
    collect_predictions()
autoplot(svm_rbf_tuned)

# select best model
svm_rbf_tuned_best <- svm_rbf_tuned %>% 
    select_best("roc_auc")
svm_rbf_tuned_best

# prep data to get the AUC and accuracy metrics
svm_rbf_train_preds <- 
    svm_rbf_tuned %>% 
    collect_predictions(parameters = svm_rbf_tuned_best)

# create tibble and add row to model metrics
svm_rbf_train_metrics_tbl <- tibble(
    model = "SVM RBF",
    sample = "CV",
    roc_auc = svm_rbf_train_preds %>%
        roc_auc(Credit_Application_Result, .pred_Creditworthy) %>%
        pull(.estimate) %>% round( . , 3),
    accuracy = svm_rbf_train_preds %>%
        accuracy(Credit_Application_Result, .pred_class) %>%
        pull(.estimate) %>% round( . , 3)
)

# create a tibble for plotting
svm_rbf_train_tbl <- 
    svm_rbf_train_preds %>% 
    roc_curve(Credit_Application_Result, .pred_Creditworthy) %>% 
    mutate(model = str_c(
        "SVM RBF - train - ",
        "AUC: ", svm_rbf_train_metrics_tbl$roc_auc,
        " - ",
        "Accuracy: ", svm_rbf_train_metrics_tbl$accuracy))

# plot
print_roc_curve( svm_rbf_train_tbl )


#### compare ----

# ROC Curve
print_roc_curve( bind_rows(logistic_train_tbl, 
                           rand_forest_train_tbl,
                           svm_rbf_train_tbl) )


## Test ----

### logistic regression ----

logistic_test_wflow <- logistic_wflow %>% 
    finalize_workflow(logistic_tuned_best)

set.seed(123)
logistic_test_fit <- 
    logistic_test_wflow %>% 
    last_fit(credit_split)

# get the AUC and accuracy metrics
logistic_test_preds <- 
    logistic_test_fit %>% 
    collect_predictions()

# create tibble and add row to model metrics
logistic_test_metrics_tbl <- tibble(
    model = "Logistic Regression",
    sample = "Test",
    roc_auc = logistic_test_preds %>%
        roc_auc(Credit_Application_Result, .pred_Creditworthy) %>%
        pull(.estimate) %>% round( . , 3),
    accuracy = logistic_test_preds %>%
        accuracy(Credit_Application_Result, .pred_class) %>%
        pull(.estimate) %>% round( . , 3)
)

# create plot label

# get the AUC
logistic_test_auc <- 
    logistic_test_metrics_tbl %>% 
    #filter(model == "Logistic Regression" & sample == "Test") %>% 
    pull(roc_auc)
# build label
logistic_test_label <- 
    str_c("Logistic Reg - test - ", 
          " - AUC: ", logistic_test_metrics_tbl$roc_auc,
          " - ",
          "Accuracy: ", logistic_test_metrics_tbl$accuracy)

# create a tibble for plotting
logistic_test_tbl <- 
    logistic_test_fit %>% 
    collect_predictions() %>% 
    roc_curve(Credit_Application_Result, .pred_Creditworthy) %>% 
    mutate(model = logistic_test_label)

# plot
print_roc_curve( logistic_test_tbl )


### random forest ----

rand_forest_test_wflow <- rand_forest_wflow %>% 
    finalize_workflow(rand_forest_tuned_best)

set.seed(123)
rand_forest_test_fit <- 
    rand_forest_test_wflow %>% 
    last_fit(credit_split)

# get the AUC and accuracy metrics
rand_forest_test_preds <- 
    rand_forest_test_fit %>% 
    collect_predictions()

# create tibble and add row to model metrics
rand_forest_test_metrics_tbl <- tibble(
    model = "Random Forest",
    sample = "Test",
    roc_auc = rand_forest_test_preds %>%
        roc_auc(Credit_Application_Result, .pred_Creditworthy) %>%
        pull(.estimate) %>% round( . , 3),
    accuracy = rand_forest_test_preds %>%
        accuracy(Credit_Application_Result, .pred_class) %>%
        pull(.estimate) %>% round( . , 3)
)

# define plot label

# get the AUC
rand_forest_test_auc <- 
    rand_forest_test_metrics_tbl %>% 
    #filter(model == "Random Forest" & sample == "Test") %>% 
    pull(roc_auc)
# build label
rand_forest_test_label <- 
    str_c("Random Forest - test - ", 
          " - AUC: ", rand_forest_test_metrics_tbl$roc_auc,
          " - ",
          "Accuracy: ", rand_forest_test_metrics_tbl$accuracy)

# create a tibble for plotting
rand_forest_test_tbl <- 
    rand_forest_test_fit %>% 
    collect_predictions() %>% 
    roc_curve(Credit_Application_Result, .pred_Creditworthy) %>% 
    mutate(model = rand_forest_test_label)

# plot
print_roc_curve( rand_forest_test_tbl )


### support vm ----

svm_rbf_test_wflow <- svm_rbf_wflow %>% 
    finalize_workflow(svm_rbf_tuned_best)

set.seed(123)
svm_rbf_test_fit <- 
    svm_rbf_test_wflow %>% 
    last_fit(credit_split)

# get the AUC and accuracy metrics
svm_rbf_test_preds <- 
    svm_rbf_test_fit %>% 
    collect_predictions()

# create tibble and add row to model metrics
svm_rbf_test_metrics_tbl <- tibble(
    model = "SVM RBF",
    sample = "Test",
    roc_auc = svm_rbf_test_preds %>%
        roc_auc(Credit_Application_Result, .pred_Creditworthy) %>%
        pull(.estimate) %>% round( . , 3),
    accuracy = svm_rbf_test_preds %>%
        accuracy(Credit_Application_Result, .pred_class) %>%
        pull(.estimate) %>% round( . , 3)
)

# define plot label

# get the AUC
svm_rbf_test_auc <- 
    svm_rbf_test_metrics_tbl %>% 
    #filter(model == "Random Forest" & sample == "Test") %>% 
    pull(roc_auc)
# build label
svm_rbf_test_label <- 
    str_c("SVM RBF - test - ", 
          " - AUC: ", svm_rbf_test_metrics_tbl$roc_auc,
          " - ",
          "Accuracy: ", svm_rbf_test_metrics_tbl$accuracy)

# create a tibble for plotting
svm_rbf_test_tbl <- 
    svm_rbf_test_fit %>% 
    collect_predictions() %>% 
    roc_curve(Credit_Application_Result, .pred_Creditworthy) %>% 
    mutate(model = svm_rbf_test_label)

# plot
print_roc_curve( svm_rbf_test_tbl )


#### compare ----

# ROC Curve
print_roc_curve( bind_rows(logistic_test_tbl, 
                           rand_forest_test_tbl,
                           svm_rbf_test_tbl) )


# Metric tibble
models_metric_tbl <- bind_rows(logistic_train_metrics_tbl,
                               logistic_test_metrics_tbl,
                               rand_forest_train_metrics_tbl,
                               rand_forest_test_metrics_tbl,
                               svm_rbf_train_metrics_tbl,
                               svm_rbf_test_metrics_tbl)


# Finalize ----

# winner: random forest

final_wflow <- 
    rand_forest_test_wflow %>% 
    finalize_workflow(rand_forest_tuned_best)

set.seed(123)
final_fit <- 
    final_wflow %>% 
    last_fit(credit_split)


# Predict ----

# get the workflow
prediction_wflow <- extract_workflow(final_fit)

# fit workflow & testing data
set.seed(123)
predict_fit <- 
    fit(prediction_wflow, testing(credit_split))

# predict for customers_to_score
new_customers <- augment(predict_fit,
                         new_data = customers_to_score) %>% 
    select(Customer_ID, starts_with(".pred"), everything()) %>% 
    arrange(desc(.pred_Creditworthy))

# plot the 2 creditworthy classes
new_customers_plot <- new_customers %>% 
    select(.pred_class) %>%
    ggplot( aes(x = .pred_class) ) +
    geom_bar(alpha = 0.25) +
    geom_text( aes(label = ..count..), 
               stat = 'count',
               vjust = 1) +
    theme_light() +
    labs(title = "New Customers Totals",
         subtitle = "Credit Worthiness",
         caption = "source: 'customers-to-score.xlsx'   ",
         x = "", y = "")
new_customers_plot

# bin the customers by percentage
new_customers_binned_plot <- new_customers %>% 
    select(starts_with(".pred")) %>% 
    mutate(bin = case_when(
        .pred_Creditworthy >= 0.90 ~ "90-100%",
        .pred_Creditworthy >= 0.80 ~ "80-90%",
        .pred_Creditworthy >= 0.70 ~ "70-80%",
        .pred_Creditworthy >= 0.60 ~ "60-70%",
        .pred_Creditworthy >= 0.50 ~ "50-60%",
        .pred_Creditworthy >= 0.40 ~ "40-50%",
        .pred_Creditworthy >= 0.30 ~ "30-40%",
        .pred_Creditworthy >= 0.20 ~ "20-30%",
        .pred_Creditworthy >= 0.10 ~ "10-20%",
        TRUE ~ "< 10%"
        )
    ) %>% 
    group_by(bin) %>% 
    ggplot(aes(x = bin)) +
    geom_bar(alpha = 0.25) +
    geom_text( aes(label = ..count..), 
               stat = 'count',
               vjust = 1) +
    theme_light() +
    labs(title = "New Customers Binned",
         subtitle = "Credit Worthiness",
         caption = "source: 'customers-to-score.xlsx'   ",
         x = "", y = "")
new_customers_binned_plot

# VIP plot
vip_plot <- prediction_wflow %>% 
    extract_fit_parsnip() %>% 
    vip(num_features = 10,
        geom = c("point"))
vip_plot <- vip_plot + 
    theme_light() +
    labs(title = "Variable Importance Plot",
         caption = "source: 'customers-to-score.xlsx'   ")
vip_plot

# write the prediction to a file
new_customers %>% 
    write_csv("../New Customers Credit Worthiness.csv")

    
