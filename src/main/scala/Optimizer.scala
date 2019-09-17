import breeze.linalg._
import breeze.stats.distributions._
import math.Pi
import breeze.numerics._
import dk.gp.gpr._
import breeze.optimize._

class OptimizerResult(val func_val : Double,
                       val x : DenseVector[Double],
                       val pastevals : DenseMatrix[Double],
                       val scores : DenseVector[Double]) {
  override def toString(): String = {
    val objective_calls = scores.length
    s"func_val : $func_val \nx : $x \nnumber of objective calls: $objective_calls\n"
  }
        }


trait Optimizer {
  def apply(acquisition: Acquisition,
            gp: GprModel,
            past_evals: DenseVector[Double],
            domain : Domain,
            lower_bounds: DenseVector[Double],
            upper_bounds: DenseVector[Double]): DenseVector[Double]
}

object LBFGSB extends Optimizer {
  def apply(acquisition: Acquisition,
            gp: GprModel,
            past_evals: DenseVector[Double],
            domain : Domain,
            lower_bounds: DenseVector[Double],
            upper_bounds: DenseVector[Double]): DenseVector[Double] = {
      //need to define lower bounds and upper bounds
      def toOptimize(z: DenseVector[Double]): Double = {
        if (z.forall((i, v) => (lower_bounds(i) <= v && v <= upper_bounds(i))))
          -acquisition.eval(z, gp, past_evals)
        else
          0 }
      val diff = new ApproximateGradientFunction(toOptimize)
      val acq  = new DiffFunction[DenseVector[Double]] {
        def calculate(z: DenseVector[Double]) = (toOptimize(z),diff.gradientAt(z))
      }
      val lbfgs     = new LBFGS[DenseVector[Double]](100,4)
      //val lbfgsb = new LBFGSB(lower_bounds,upper_bounds)
      val initials = for (i <- 0 until lower_bounds.length)
       yield Uniform(lower_bounds(i),upper_bounds(i)).draw
      val init = DenseVector(initials:_*)
      val optimum = lbfgs.minimize(acq,init)
      //println(optimum)
      domain.correctSample(optimum)
  }
}