library(data.table)
library(inspectdf)
library(dplyr)
library(ranger)
library(caret)
library(ggplot2)

#----- Cargo datos
datLabel <- as.data.frame(fread("01_labels.csv", nThread = 4 ))

datTrain <- as.data.frame(fread("02_trainset.csv", nThread = 4 ))

datos <- merge(datTrain, datLabel, by.x = "id", by.y = "id", all = TRUE)

head(datos)

save(datos, file = "datosRaw.RData")

datos %>%
  select(status_group) %>%
  group_by(status_group) %>%
  summarise( f_pump = n()) %>%
  ggplot( aes( status_group, f_pump)) +
  geom_col( fill = 'blue')

# Como se haria con data.table
datDT <- as.data.table(datos)
#datDT[filas, columnas, groupby]
datDT[, .(f_pump = .N), by=.(status_group)]

summary(datos)

as.data.frame(mapply(class, datos))
# Ver que tiene basin, subvillage, region
as.data.frame(table(datos$basin))
as.data.frame(table(datos$subvillage))
as.data.frame(table(datos$region))

# Distribuci√≥n de basin
datos %>% 
  select(basin) %>% 
  group_by(basin) %>%
  summarize( frecu = n()) %>%
  arrange( desc(frecu))  %>%
  ggplot( aes(basin, frecu)) +
  geom_col()

# Distribuci√≥n de subvillage
dat_subv <- datos %>% 
  select(subvillage) %>% 
  group_by(subvillage) %>%
  summarize( frecu = n() ) %>%
  arrange( desc(frecu)) 


dat_subv %>%
  filter(frecu > 100)

# Distribuci√≥n de region
datos %>% 
  select(region) %>% 
  group_by(region) %>%
  summarize( frecu = n() ) %>%
  arrange( desc(frecu))


mycol <- 'funder'
datos %>% 
  select(!!sym(mycol)) %>% 
  group_by(!!sym(mycol)) %>%
  summarize( frecu = n() ) %>%
  arrange( desc(frecu))

# categorical plot
x <- inspect_cat(datos) 
show_plot(x)

# correlations in numeric columns
x <- inspect_cor(datos)
show_plot(x)

# feature imbalance bar plot
x <- inspect_imb(datos)
show_plot(x)

# memory usage barplot
x <- inspect_mem(datos)
show_plot(x)

# missingness barplot
x <- inspect_na(datos)
show_plot(x)

# histograms for numeric columns
x <- inspect_num(datos)
show_plot(x)

# barplot of column types
x <- inspect_types(datos)
show_plot(x)

#------ Tranformao las variables logicas con 0 / 1... Son las que tienen NAs.... 
# Las variables que mas NAs tienen son las logic -> public_meeting + permit - lo veo con inspectdf
# Las codifico primeramente a numeric y luego las selecciono para imputarlas
datos$fe_publicmeeting <- ifelse(datos$public_meeting == TRUE, 1, 0)
datos$fe_permit        <- ifelse(datos$permit == TRUE, 1, 0)


# Cogemos las  numericas solo
var_tmp <- mapply(class, datos)

var_tp <- as.data.frame(mapply(class, datos))
names(var_tp)[1] <- 'tipo'
var_tp$myvar <-  rownames(var_tp)
rownames(var_tp) <- NULL


# Compruebo que efectivamente no he eliminado ning√∫n NA...
sum(is.na(datos$fe_permit))

sum(is.na(datos$fe_publicmeeting))

sum(is.na(datos$public_meeting))

sum(is.na(datos$permit))



var_nums <- var_tp[var_tp$tipo == 'integer' | var_tp$tipo == 'numeric',]
var_nums

datos_num <- datos[ , c(var_nums$myvar, "status_group")] # Incluyo status_group
names(datos_num)
datos_num$status_group <- as.factor(datos$status_group)


# missRanger ----------------------------------------------
# Ahora voy a imputar esas dos variables con missRanger
library(missRanger)
datEnnum_imp <- missRanger(
  datos_num, 
  fe_publicmeeting + fe_permit ~. ,
  pmm = 3
)

sum(is.na(datEnnum_imp$fe_permit))

sum(is.na(datEnnum_imp$fe_publicmeeting))

#--- Clustimpute

library(ClustImpute)
library(tictoc)
datos_num$status_group <- NULL # En este caso tengo que eliminar la "target" - Solo numericas.
nr_iter    <- 10                # iterations of procedure
n_end      <- 10                # step until convergence of weight function to 1
nr_cluster <- 3                 # number of clusters - Los tipos del estado de las bombas que tengo que predecir
c_steps    <- 50                # numer of cluster steps per iteration
tic("Run ClustImpute")
datos_clusimpute <- ClustImpute(datos_num,nr_cluster=nr_cluster, nr_iter=nr_iter, c_steps=c_steps, n_end=n_end) 
toc()

#--------- H2O 
library(h2o)
# h2o.init() # Forma sencilla de arrancar - Ning˙n par·metro.

h2o.init(nthreads = 4, max_mem_size = '2g') # Le doy CPU + Memoria
# To speedup transformations
options("h2o.use.data.table" = TRUE)
h2o.no_progress()

datIn_hex <- as.h2o(datEnnum_imp)

splits <- h2o.splitFrame( 
  data = datIn_hex, 
  ratios = c(0.6,0.2), 
  ## only need to specify 2 fractions, the 3rd is implied 
  destination_frames = c("train_hex", "valid_hex", "test_hex"), 
  seed = 1234
) 
train_hex <- splits[[1]] 
valid_hex <- splits[[2]] 
test_hex  <- splits[[3]]

# val_trees <- c(50,100,150,200)
# val_mxdep <- c(5, 10, 15, 20)
# par_grid  <- expand.grid(val_trees, val_mxdep)
# head(par_grid)

h2o.no_progress()
# Identify predictors and response
y <- "status_group"
x <- setdiff(names(datIn_hex), y)
train_hex[, y] <- as.factor( train_hex[,y] )

nfolds <- 5

val_trees <- c(50,100,150,200)
val_mxdep <- c(5, 10, 15, 20)

par_grid  <- expand.grid(val_trees, val_mxdep)

res_df <- data.frame()
for (i in 1:nrow(par_grid)) {
  
  tmp_tree <- par_grid[i, 1]
  tmp_mxde <- par_grid[i, 2]
  
  my_modelel <- h2o.randomForest(
    x = x,
    y = y,
    training_frame = train_hex,
    validation_frame = valid_hex,
    nfolds = nfolds,
    keep_cross_validation_predictions = TRUE,
    seed = 3436467, 
    stopping_metric = 'AUC',
    verbose = FALSE,
    ntrees = tmp_tree,
    max_depth = tmp_mxde
  )
  
  my_auc <- h2o.auc(h2o.performance(my_modelel, newdata = test_hex))
  
  # Guardo resultados de cada ejecuciÛn 
  tmp_vec <- c(tmp_tree, tmp_mxde, my_auc)
  res_df  <- rbind(res_df, tmp_vec)
  print(res_df)
  
} # for(i in)---------

# Cambio nombres de df resultante
names(res_df) <- c('ntree', 'mxdep' ,'mi_auc')
res_df

# Combinacion ganadora
res_df[ res_df$mi_auc == max(res_df$mi_auc), ]

my_gr <- ggplot(res_df, aes(x = ntree, y = mxdep)) +
  geom_point( aes(size = mi_auc*10 )) +
  theme_bw()
print(my_gr)

# Submission
test <- as.data.frame(fread('Test_set_values.csv', nThread = 3))
pred_test <- as.vector(predict(my_model, test)$prediction)

my_sub <- data.frame(
  id = test$id,
  status_group = pred_test
)
fwrite(my_sub, file= "sub_v02_base_h2o.csv", sep=",")



