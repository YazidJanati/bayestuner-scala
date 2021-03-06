package com.github.yazidjanati
import breeze.linalg._
import math._
import breeze.stats.distributions._


/** Domain models the input space of the objective.
 *
 *   @constructor     creates a new domain.
 *   @param  bounds   an array containing each dimension of the input space.
 *                    A dimension is characterized by a lower bound, an upper
 *                    bound and a type. Type is either continuous or discrete.
 *                    if it's discrete, the optimization is done on a grid of
 *                    integers within the bounds.
 */
class Domain(val bounds : Array[(Double,Double,String)]) {

  /** Processes each sample generated by the algorithm to make sure it has the
   *  correct type (continuous or discrete) and that it is within the bounds.
   *
   *  @param sample    the vector to process.
   */

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
