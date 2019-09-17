import breeze.linalg._
import dk.gp.gpr._
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics._
import breeze.math._
import breeze.generic.UFunc


//Iterative Bayesian Optimization with hard bounds
object IBOhardBounds extends Model {

  def apply(obj: DenseMatrix[Double] => DenseMatrix[Double],
            bounds: DenseMatrix[Double],
            n_samples: Int,
            n_subiter: Int,
            power : Double,
            n_iter: Int = 10,
            covFunction: CovFunc = CovSEiso(),
            covFuncParams: DenseVector[Double] = DenseVector(log(1d), log(1)),
            noise:  Double = 0,
            discrete_inputs: Boolean = false
            ): DenseMatrix[Double] = {

      val initial_bounds =  bounds
      var bounds_        =  bounds

      var curr_optim     =  DenseMatrix.zeros[Double](1,bounds.rows)

      for (i <- 0 to n_iter) {
        curr_optim  = Bayesopt(obj,bounds_,n_samples,n_subiter,covFunction,covFuncParams,noise,discrete_inputs)

        require(check_within_bounds(curr_optim.toDenseVector,bounds_),"Not within boundssss")

        bounds_     = reduce_bounds(curr_optim,initial_bounds,i,power)
      }

      curr_optim
            }

  def reduce_bounds(center : DenseMatrix[Double],
                    bounds: DenseMatrix[Double],
                    iteration: Int,
                    power : Double) : DenseMatrix[Double] = {

      val lower  =   for (i <- 0 until bounds.rows)
                    yield max(bounds(i,0),center(0,i) - 4 / (pow(2,pow(iteration.toDouble,1.0/power))))
      val upper  =   for (i <- 0 until bounds.rows)
                    yield min(bounds(i,1),center(0,i) + 4 / (pow(2,pow(iteration.toDouble,1.0/power))))
      val bound  =   for (i <- 0 until bounds.rows)
                    yield (lower(i),upper(i))

      DenseMatrix(bound:_*)
      }

  def total_evaluations(n_samples: Int, n_iter: Int, n_subiter: Int): Int = {
    Bayesopt.total_evals(n_samples,n_subiter) * (n_iter + 1)
  }

}
