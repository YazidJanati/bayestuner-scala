import breeze.linalg._
import math._
import breeze.stats.distributions._


class Domain(val bounds : Array[(Double,Double,String)]) {
  //require(Array("continuous","discrete") contains bounds._3)

  def correctSample(sample : DenseVector[Double]) : DenseVector[Double] = {
    require(sample.length == this.bounds.length,"sample length must be equal to space dimension")
    var correctedSmple = Array.empty[Double]
    for (x <- sample.data zip this.bounds.map( y => y._3)) {
      if (x._2 == "continuous")
        correctedSmple = correctedSmple :+ x._1
      else if (x._2 == "discrete")
        correctedSmple = correctedSmple :+ math.round(x._1).toDouble
    }
    DenseVector(correctedSmple:_*)  }

}
