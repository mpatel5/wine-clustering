# Import Libraries
library(readr)
library(cluster)
library(factoextra)

# Import Dataset
df <- read_csv("https://raw.githubusercontent.com/mpatel5/wine-clustering/main/wine-clustering.csv")

# Data Preview
head(df)
str(df)
summary(df)

# Data Quality Checks
sum(is.na(df))
sum(duplicated(df))
df[duplicated(df), ]

# Outlier Detection
numeric_df <- df[, sapply(df, is.numeric)]
z_scores <- scale(numeric_df)
sum(abs(z_scores) > 3, na.rm = TRUE)

df_clean <- df[!apply(abs(z_scores) > 3, 1, any), ]

# Train Test Split
set.seed(123)
n <- nrow(df_clean)
train_index <- sample(1:n, size = 0.8 * n)

train_data <- df_clean[train_index, ]
test_data  <- df_clean[-train_index, ]

# Scaling
train_numeric <- train_data[, sapply(train_data, is.numeric)]
test_numeric  <- test_data[, sapply(test_data, is.numeric)]

train_scaled <- scale(train_numeric)

test_scaled <- scale(
  test_numeric,
  center = attr(train_scaled, "scaled:center"),
  scale  = attr(train_scaled, "scaled:scale")
)

# Elbow Method
wss <- numeric(10)

for (k in 1:10) {
  set.seed(123)
  km <- kmeans(train_scaled, centers = k, nstart = 25)
  wss[k] <- km$tot.withinss
}

plot(1:10, wss, type = "b", pch = 19,
     xlab = "Number of Clusters (k)",
     ylab = "Total Within-Cluster Sum of Squares",
     main = "Elbow Method")

abline(v = 3, col = "red", lty = 2)
text(3, wss[3], labels = "k = 3", pos = 4, col = "red")

# K-means Model
set.seed(123)
kmeans_model <- kmeans(train_scaled, centers = 3, nstart = 25)

# Training Results
train_clusters <- kmeans_model$cluster
train_wss <- kmeans_model$tot.withinss

train_sil <- silhouette(train_clusters, dist(train_scaled))
avg_train_sil <- mean(train_sil[, 3])

cat("Training WSS:", train_wss, "\n")
cat("Training Silhouette:", avg_train_sil, "\n")
cat("Training Cluster Sizes:\n")
print(table(train_clusters))

# Test Assignment
centers <- kmeans_model$centers

assign_cluster <- function(x, centers) {
  distances <- apply(centers, 1, function(center) sum((x - center)^2))
  which.min(distances)
}

test_clusters <- apply(test_scaled, 1, assign_cluster, centers = centers)

# Test Results
test_sil <- silhouette(test_clusters, dist(test_scaled))
avg_test_sil <- mean(test_sil[, 3])

cat("Test Silhouette:", avg_test_sil, "\n")
cat("Test Cluster Sizes:\n")
print(table(test_clusters))

# Visualization
plot(train_scaled[,1], train_scaled[,2],
     col = train_clusters,
     pch = 19,
     xlab = colnames(train_scaled)[1],
     ylab = colnames(train_scaled)[2],
     main = "Training Clusters")

plot(test_scaled[,1], test_scaled[,2],
     col = test_clusters,
     pch = 19,
     xlab = colnames(test_scaled)[1],
     ylab = colnames(test_scaled)[2],
     main = "Test Clusters")