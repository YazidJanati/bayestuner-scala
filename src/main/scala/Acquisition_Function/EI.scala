import breeze.linalg._
import breeze.stats.distributions._
import math.Pi
import breeze.numerics._
import dk.gp.gpr._
import breeze.optimize._
import math.ceil
import math.floor

object EI extends Model {

  def calculate_EI(z: DenseMatrix[Double],
                  bounds: DenseMatrix[Double],
                  y_sample: DenseVector[Double],
                  gpr : GprModel): Double =  {
      val lower_bounds = bounds(::,0)
      val upper_bounds = bounds(::,1)

      val prediction =  gprPredict(z, gpr)
      val mean       =  prediction(::,0)
      val variance   =  prediction(::,1)

      val std_gaussian   =  Gaussian(0,1)
      val std_pdf        =  (x: Double) => exp(-math.pow(x,2) / 2) / sqrt(2*Pi)

      if (z.toDenseVector.forall((i, v) => (lower_bounds(i) <= v && v <= upper_bounds(i))) == false) 0

      else if (variance(0) == 0) 0

      else {
        val y_max   = y_sample.max
        val delta   = mean(0) - y_max
        val Z       = delta / variance(0)
        val Z_      = Z
        delta * std_gaussian.cdf(Z) + variance(0) * std_pdf(Z_)
        }
      }

  def EI_optim(x_sample : DenseMatrix[Double],
                y_sample: DenseVector[Double],
                gpr: GprModel,
                bounds: DenseMatrix[Double],
                discrete_inputs : Boolean): DenseMatrix[Double] = {

      val lower_bounds  = bounds(::,0)
      val upper_bounds  = bounds(::,1)

      def surrogate(z: DenseVector[Double]): Double = -calculate_EI(z.toDenseMatrix, bounds,
                                                  y_sample,gpr)
      val diff_obj  = new ApproximateGradientFunction(surrogate)
      val ei        = new DiffFunction[DenseVector[Double]] {
        def calculate(z: DenseVector[Double]) = (surrogate(z), diff_obj.gradientAt(z))
      }
      //val lbfgsb    = new LBFGSB(lower_bounds,upper_bounds)
      val lbfgs     = new LBFGS[DenseVector[Double]](100,4)

      var u         = for (i <- 0 until lower_bounds.length)
       yield Uniform(lower_bounds(i),upper_bounds(i)).draw
      val init    = DenseVector(u:_*)
      val optimum = lbfgs.minimize(ei,init)
      optimum.toDenseMatrix
    }

  def next_eval(x_sample : DenseMatrix[Double],
                y_sample: DenseVector[Double],
                gpr: GprModel,
                bounds: DenseMatrix[Double],
                discrete_inputs: Boolean): DenseMatrix[Double] = {

      var optimum = EI_optim(x_sample,y_sample,gpr,bounds,discrete_inputs).toDenseVector
      optimum     = if (discrete_inputs) vector_rounded(optimum) else optimum
      if (check_within_bounds(optimum, bounds))
        optimum.toDenseMatrix
      else next_eval(x_sample,y_sample,gpr,bounds,discrete_inputs)
    }

    }
