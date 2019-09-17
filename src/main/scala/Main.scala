import dk.gp.gpr._
import dk.gp.cov.CovSEiso
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.stats.distributions._
import breeze.optimize._
import math._

object Main extends App {
  val bounds = Array((- 10d,10d,"continuous"),(- 10d,10d,"continuous"))
  def obj(x: DenseVector[Double]): Double = {
    - math.pow(x(0),2) - math.pow(x(1),2)
  }

  def rastringin(x: DenseVector[Double]): Double = {
    val sum = x.map(y => y*y - 10*math.cos(2*math.Pi*y)).reduceLeft(_+_)
    -10*x.length - sum
  }
  val bayesiantuner = new BayesianTuner(rastringin,
                                        bounds,
                                        80,15)
  val optim = bayesiantuner.tune
  println(optim)
  /**val dom = new Domain(Array((- 10,10,"continuous"),(- 10,10,"discrete")))
  val lower_bounds   = DenseVector(dom.bounds.map(x => x._1))
  val upper_bounds   = DenseVector(dom.bounds.map(x => x._2))
  println(dom.genSamples(10,"normal",lower_bounds,upper_bounds))*/
}
