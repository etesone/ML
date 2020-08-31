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

# Distribución de basin
datos %>% 
  select(basin) %>% 
  group_by(basin) %>%
  summarize( frecu = n()) %>%
  arrange( desc(frecu))  %>%
  ggplot( aes(basin, frecu)) +
  geom_col()

# Distribución de subvillage
dat_subv <- datos %>% 
  select(subvillage) %>% 
  group_by(subvillage) %>%
  summarize( frecu = n() ) %>%
  arrange( desc(frecu)) 


dat_subv %>%
  filter(frecu > 100)

# Distribución de region
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


# Compruebo que efectivamente no he eliminado ningún NA...
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

# library(dataPreparation)
# 
# prepareSet( datos_clusimpute, finalForm = 'data.table', key = 'status_group', target_col = 'status_group')

datos323<-merge(datos_clusimpute$complete_data,datLabel, by.x = "id", by.y = "id", all = TRUE)

datos323$status_group <- as.factor(datos323$status_group)

library(ranger)
my_mod <- ranger( 
  as.factor(status_group) ~ id + amount_tsh + longitude + latitude + gps_height + num_private + region_code + district_code + population + construction_year,
  data = datos323, importance = 'impurity',
  verbose = TRUE, min.node.size = 1,
  num.trees = 1000
)


pred_mod <- predict(my_mod, datos323)

confusionMatrix( pred_mod$predictions, as.factor(datos323$status_group))

rang_imp <- as.data.frame(importance(my_mod))
rang_imp$vars <- rownames(rang_imp)
names(rang_imp)[1] <- c('importance')
rownames(rang_imp) <- NULL
rang_imp %>%
  arrange(desc(importance))

# Submission
test <- as.data.frame(fread('Test_set_values.csv', nThread = 3))
pred_test <- as.vector(predict(my_mod, test)$prediction)

my_sub <- data.frame(
  id = test$id,
  status_group = pred_test
)
fwrite(my_sub, file= "sub_v02_base_Clust_Impute1.csv", sep=",")



