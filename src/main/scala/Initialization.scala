import breeze.linalg._
import math._
import breeze.stats.distributions._


/** The gaussian process regression needs to be initialized with some points in
    the input space. This class models the initialization strategy.
 */

abstract class Initialization {

  /** Generated a certain number of samples.
   *
   *  @param domain       The input space of the objective.
   *  @param num_samples  Number of points to sample.
   *  @return Vector of samples.
   *
   */

  def apply(domain : Domain,
            num_samples : Int) : DenseMatrix[Double]

  /** Since we can't choose bounds for certain probability distributions, we
   *  need to make sure that our samples are within the input space.
   *
   *  @param sample Sample to investigate.
   *  @param domain Input space of the objective.
   *  @return whether the sample is within the bounds.
   */

  def check_within_bounds(sample: DenseVector[Double],
                          domain : Domain) : Boolean = {

      (sample.data zip domain.bounds.map(x => (x._1,x._2))).forall(
          value => value._2._1  < value._1 && value._1 < value._2._2)
    }
}

/** Initialization using a gaussian distribution.
 */

object NormalInit extends Initialization {

  def apply(domain: Domain,
               num_samples : Int) : DenseMatrix[Double] = {
      val means = DenseVector((for (bound <- domain.bounds)
                                  yield (bound._1 + bound._2)/2):_*)
      val cov   = diag(DenseVector((domain.bounds).map(x => (x._2 - x._1)/2)))
      val rawSamples = for (i <- 0 until num_samples)
                           yield MultivariateGaussian(means,cov).draw()
      //println(DenseMatrix(rawSamples:_*))
      DenseMatrix(rawSamples:_*)
               }
}

/** Initialization using a uniform distribution.
 */

object UniformInit extends Initialization {

  def apply(domain : Domain,
               num_samples : Int): DenseMatrix[Double] = {
      val lower_bounds = domain.bounds.map(x => x._1)
      val upper_bounds = domain.bounds.map(x => x._2)
      val rawSamples   = for (i <- 0 until num_samples)
                          yield {for (j <- 0 until lower_bounds.length)
                                yield Uniform(lower_bounds(j),upper_bounds(j)).draw()}
      DenseMatrix(rawSamples.map( x => domain.correctSample(DenseVector(x:_*))).toArray:_*)
               }
}
