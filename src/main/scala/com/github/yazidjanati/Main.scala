package com.github.yazidjanati
import dk.gp.gpr._
import dk.gp.cov.CovSEiso
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.stats.distributions._
import breeze.optimize._
import math._

object Main extends App {
  val bounds = Array((- 10.0,10.0,"continuous"),(- 10.0,10.0,"continuous"))
  def obj(x: DenseVector[Double]): Double = {
    - math.pow(x(0),2) - math.pow(x(1),2)
  }
  val bayesiantuner = new BayesianTuner(obj,
                                        bounds,
                                        40,15)
  val optim = bayesiantuner.tune
  /**val dom = new Domain(Array((- 10,10,"continuous"),(- 10,10,"discrete")))
  val lower_bounds   = DenseVector(dom.bounds.map(x => x._1))
  val upper_bounds   = DenseVector(dom.bounds.map(x => x._2))
  println(dom.genSamples(10,"normal",lower_bounds,upper_bounds))*/
}
