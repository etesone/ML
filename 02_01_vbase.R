
###################
# Version inicial
# Fecha : xcxxxxx
# Objetivo: Version base
# Resultado: 
#################

#------Cargo librerias
library(data.table)
library(inspectdf)
library(dplyr)
library(ranger)
library(caret)

# #----- Cargo datos
# datLabel <- as.data.frame(fread("01_labels.csv", nThread = 4 ))
# dim(datLabel)
# head(datLabel)
# datTrain <- as.data.frame(fread("02_trainset.csv", nThread = 4 ))
# dim(datTrain)
# head(datTrain)
# 
# #----- Junto labels y Train
# head(datLabel[,1])
# head(datTrain[,1])
# all.equal(datLabel[,1], datTrain[,1])
# # tienen el mismo orden pero aun asi hago un merge.
# 
# datEnd <- merge(datTrain, datLabel, by.x = "id", by.y="id", all = TRUE)
# head(datEnd)

# save(datEnd, file = "datosRaw.RData")
load("datosRaw.RData")

#------Explorar datos

# Estudio la target 'status_group'
datEnd %>%
  select(status_group) %>%
  group_by(status_group) %>%
  summarise( f_pump = n()) %>%
  ggplot( aes( status_group, f_pump)) +
   geom_col( fill = 'blue')

# Como se haria con data.table
datDT <- as.data.table(datEnd)
#datDT[filas, columnas, groupby]
datDT[, .(f_pump = .N), by=.(status_group)]

summary(datEnd)

as.data.frame(mapply(class, datEnd))
# Ver que tiene basin, subvillage, region
as.data.frame(table(datEnd$basin))
as.data.frame(table(datEnd$subvillage))

# Estudio variables con dplyr
library(dplyr)
library(ggplot2)
# Distribución de basin
datEnd %>% 
  select(basin) %>% 
  group_by(basin) %>%
  summarize( frecu = n()) %>%
  arrange( desc(frecu))  %>%
  ggplot( aes(basin, frecu)) +
  geom_col()

# Distribución de subvillage
dat_subv <- datEnd %>% 
  select(subvillage) %>% 
  group_by(subvillage) %>%
  summarize( frecu = n() ) %>%
  arrange( desc(frecu)) 


dat_subv %>%
  filter(frecu > 100)

# Distribución de region
datEnd %>% 
  select(region) %>% 
  group_by(region) %>%
  summarize( frecu = n() ) %>%
  arrange( desc(frecu))


mycol <- 'funder'
datEnd %>% 
  select(!!sym(mycol)) %>% 
  group_by(!!sym(mycol)) %>%
  summarize( frecu = n() ) %>%
  arrange( desc(frecu))

#------------------------
# Numeric variables 
vars_df <- as.data.frame(mapply(class, datEnd))
vars_df$vars <- rownames(vars_df)
rownames(vars_df) <- NULL
names(vars_df)[1] <- c('tipo')
vars_num <- vars_df[ vars_df$tipo %in% c('integer', 'numeric'),]


# Cuantos niveles tienen las variables categoricas
# para luego escoger unas cuantas de ellas que tengan
# pocos nivels para que ya entren en el modelo.

for (i in 1:ncol(datEnd)) {

  tmp_clas <- class(datEnd[,i])
  
  if (tmp_clas == 'character') {
     tmp_lg <- length(unique(datEnd[,i]))
     print( c(names(datEnd)[i], tmp_lg))
  } 
}
# 
# [1] "date_recorded" "356"          
# [1] "funder" "1898"  
# [1] "installer" "2146"     
# [1] "wpt_name" "37400"   
# [1] "basin" "9"    
# [1] "subvillage" "19288"     
# [1] "region" "21"    
# [1] "lga" "125"
# [1] "ward" "2092"
# [1] "recorded_by" "1"          
# [1] "scheme_management" "13"               
# [1] "scheme_name" "2697"       
# [1] "extraction_type" "18"             
# [1] "extraction_type_group" "13"                   
# [1] "extraction_type_class" "7"                    
# [1] "management" "12"        
# [1] "management_group" "5"               
# [1] "payment" "7"      
# [1] "payment_type" "7"           
# [1] "water_quality" "8"            
# [1] "quality_group" "6"            
# [1] "quantity" "5"       
# [1] "quantity_group" "5"             
# [1] "source" "10"    
# [1] "source_type" "7"          
# [1] "source_class" "3"           
# [1] "waterpoint_type" "7"              
# [1] "waterpoint_type_group" "6"                    
# [1] "status_group" "3"     

#------ Modelo
# Cogemos las  numericas solo
var_tmp <- mapply(class, datEnd)

var_tp <- as.data.frame(mapply(class, datEnd))
names(var_tp)[1] <- 'tipo'
var_tp$myvar <-  rownames(var_tp)
rownames(var_tp) <- NULL

var_tp[var_tp$tipo == 'integer' | var_tp$tipo == 'numeric',]

# tipo             myvar
# 1  integer                id
# 5  integer        gps_height
# 10 integer       num_private
# 14 integer       region_code
# 15 integer     district_code
# 18 integer        population
# 24 integer construction_year
row.names(var_tp)[ var_tp$tipo == 'integer']
# > row.names(var_tp)[ var_tp$tipo == 'integer']
# [1] "id"                "gps_height"        "num_private"      
# [4] "region_code"       "district_code"     "population"       
# [7] "construction_year"

# Creo modelo de arboles con 'ranger'

# Como no funciona en RStudio_cloud hago un sampling para 
# ver que sale...
# idx <- sample(1:nrow(datEnd), 2e4, replace = FALSE)
# datEnd_smp <- datEnd[idx, ]
# dim(datEnd_smp)

# Alguna numerica es constante?
Nuns_df <- datEnd[, c(vars_num$vars) ]
es_cte <- as.data.frame(apply(Nuns_df, 2, sd))

library(ranger)
my_mod <- ranger( 
                  as.factor(status_group) ~ id + amount_tsh + longitude + latitude + gps_height + num_private + region_code + district_code + population + construction_year,
                            data = datEnd, importance = 'impurity',
                    verbose = TRUE
                  )

                    
pred_mod <- predict(my_mod, datEnd)

confusionMatrix( pred_mod$predictions, as.factor(datEnd$status_group))

rang_imp <- as.data.frame(importance(my_mod))
rang_imp$vars <- rownames(rang_imp)
names(rang_imp)[1] <- c('importance')
rownames(rang_imp) <- NULL
rang_imp %>%
  arrange(desc(importance))


# Submission
test <- as.data.frame(fread('03_testset.csv', nThread = 3))
pred_test <- as.vector(predict(my_mod, test)$prediction)

my_sub <- data.frame(
                     id = test$id,
                     status_group = pred_test
                     )
fwrite(my_sub, file= "sub_v02_base_.csv", sep=",")


