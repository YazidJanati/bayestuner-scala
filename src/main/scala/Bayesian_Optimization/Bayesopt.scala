import breeze.linalg._
import dk.gp.gpr._
import dk.gp.cov.CovSEiso
import dk.gp.cov.CovFunc
import breeze.numerics._
import breeze.stats.distributions._
import breeze.math._
import breeze.generic.UFunc

object Bayesopt extends Model{

  def apply(obj: DenseMatrix[Double] => DenseMatrix[Double],
              bounds: DenseMatrix[Double],
              n_samples: Int,
              n_iter: Int,
              covFunction: CovFunc = CovSEiso(),
              covFuncParams: DenseVector[Double] = DenseVector(log(1d), log(1)),
              noise: Double = 0,
              discrete_inputs: Boolean = false): DenseMatrix[Double] = {

      var samples   = generate_samples(bounds,obj,n_samples,discrete_inputs)
      var x_samples = samples._1
      var y_samples = samples._2
      println(x_samples)
      println(y_samples)
      for (i <- 0 to n_iter) {
        val next_loc   = one_step(x_samples,y_samples,bounds,covFunction,covFuncParams,noise,discrete_inputs)
        println(obj(next_loc))
        x_samples      = DenseMatrix.vertcat(x_samples,next_loc)
        y_samples      = DenseVector.vertcat(y_samples,obj(next_loc).toDenseVector)
      }

      val y_argmax = argmax(y_samples)
      val optimum  = x_samples.t(::,y_argmax)

      println(optimum)
      optimum.toDenseMatrix
    }



  def generate_samples(bounds: DenseMatrix[Double],
                      obj: DenseMatrix[Double] => DenseMatrix[Double],
                      n_samples: Int,
                      discrete_inputs: Boolean) : (DenseMatrix[Double], DenseVector[Double]) = {

    val lower_bounds= bounds(::,0)
    val upper_bounds= bounds(::,1)

    val samples     = for (i <- 0 until n_samples)
                        yield { for (j <- 0 until lower_bounds.length)
                                  yield Uniform(lower_bounds(j),upper_bounds(j)).draw() }


    var x_samples   = if (discrete_inputs) matrix_rounded(DenseMatrix(samples:_*)) else DenseMatrix(samples:_*)
    var y_samples   = if (discrete_inputs) obj(matrix_rounded(DenseMatrix(samples:_*))).toDenseVector
                      else obj(DenseMatrix(samples:_*)).toDenseVector

    (x_samples,y_samples)
  }



  private def one_step(x_sample: DenseMatrix[Double],
                      y_sample: DenseVector[Double],
                      bounds: DenseMatrix[Double],
                      covFunction : CovFunc,
                      covFuncParams: DenseVector[Double],
                      noise: Double,
                      discrete_inputs: Boolean) : DenseMatrix[Double] = {
              val gpModel = gpr(x_sample,y_sample,covFunction,covFuncParams,noise)

              val next_loc= EI.next_eval(x_sample,y_sample,gpModel,bounds,discrete_inputs)

              require(check_within_bounds(next_loc.toDenseVector,bounds),"Not within bounds 1")

              next_loc.toDenseMatrix
  }

  def total_evals(n_samples: Int, n_iter: Int): Int = {
    n_samples + n_iter + 1
  }

  }
