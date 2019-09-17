import breeze.linalg._
import breeze.stats.distributions._
import math.Pi
import breeze.numerics._
import dk.gp.gpr._
import breeze.optimize._


abstract class Acquisition(i : Int,
                           temperature: Option[Int => Double]) {


  def eval(z: DenseVector[Double],
            gp: GprModel,
            past_evals : DenseVector[Double]): Double
}

class UCB(i: Int,
          temperature: Option[Int => Double])
           extends Acquisition(i: Int
                              , temperature: Option[Int => Double]) {

  val temp = temperature match {
    case None => throw new Exception("You must input a temperature function")
    case Some(f) => f
  }

  def eval(z: DenseVector[Double],
            gp: GprModel,
            past_evals: DenseVector[Double]) : Double = {
      val ztoMatrix  =  z.toDenseMatrix
      val pred       =  gprPredict(ztoMatrix, gp)
      val mean       =  pred(::,0)
      val cov        =  pred(::,1)
      val std_dev    =  math.sqrt(cov(0))

      mean(0) + temp(i) * std_dev
            }
}

class ExpectedImprovement(i: Int,
                          temperature: Option[Int => Double])
           extends Acquisition(i: Int,
                               temperature: Option[Int => Double]) {


  def eval(z: DenseVector[Double],
            gp: GprModel,
            past_evals: DenseVector[Double]): Double = {
      val ztoMatrix  = z.toDenseMatrix
      val prediction = gprPredict(ztoMatrix,gp)
      val mean       = prediction(::,0)
      val variance   = prediction(::,1)

      val std_gaussian = Gaussian(0,1)
      val gauss_pdf    = (x: Double) => exp(-math.pow(x,2) / 2) / sqrt(2*Pi)

      if (variance(0) == 0) 0
      else {
        val y_max = past_evals.max
        val delta = mean(0) - y_max
        val Z     = delta / variance(0)
        delta * std_gaussian.cdf(Z) + variance(0) * gauss_pdf(Z)
      }
            }
}
