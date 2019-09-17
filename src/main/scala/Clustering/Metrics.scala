import smile.clustering.KMeans
import scala.math

object Metrics {

  def distance(y: Array[Double],x: Array[Double]): Double = {
    require(x.size == y.size, "Arrays not of same size")
    var sum = 0.0
    for (d <- x zip y)
        sum = sum + math.pow(d._1 - d._2,2)
    math.pow(sum,1.0/2.0)
                                }
  def minimal_distance(x: Array[Double], target: Array[Array[Double]] ): Double = {
    val distances = for (point <- target)
                      yield distance(x,point)
    distances.reduceLeft(_ min _)
  }

  def closest_centroid(x: Array[Double], centroids: Array[Array[Double]]): Array[Double] = {
    var min = Double.PositiveInfinity
    var closest_ctroid = Array(0.0)
    for (centroid <- centroids) {
      if (distance(x,centroid) <= min) {
        min              = distance(x,centroid)
        closest_ctroid = centroid
      }
    }
    closest_ctroid
  }
  def distToGT(cluster: KMeans,groundtruth: Array[Array[Double]]): Double = {
    val distances_to_gt = for (centroid <- cluster.centroids)
                              yield minimal_distance(centroid,groundtruth)
    distances_to_gt.reduceLeft(_+_)
  }

  def getLabel(data: Array[Array[Double]], groundtruth: Array[Array[Double]]): Array[Int] = {
      var Labels = Map[Array[Double],Int]()
      var final_labels = Array[Int]()
      for (i <- 0 until groundtruth.length)
          Labels += (groundtruth(i) -> i)
      for (x <- data)
        final_labels = final_labels :+ Labels(closest_centroid(x,groundtruth))
      final_labels
  }

  }
