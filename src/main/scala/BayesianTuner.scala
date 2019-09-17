import breeze.linalg._
import dk.gp.gpr._
import dk.gp.cov.CovSEiso
import dk.gp.cov.CovFunc
import breeze.numerics._
import breeze.stats.distributions._
import breeze.math._
import breeze.generic.UFunc



class BayesianTuner(val obj : DenseVector[Double] => Double,
                    val bounds : Array[(Double,Double,String)],
                    val num_iter : Int,
                    val num_samples: Int,
                    val acquisition: Int => Acquisition = (i: Int) => new UCB(i, Some(i => sqrt(log(i)))),
                    val chooser: Chooser = MaxAcquisition,
                    val initialization : Initialization = UniformInit,
                    val kernel : CovFunc = CovSEiso(),
                    val kernelParams: DenseVector[Double] = DenseVector(log(1d), log(1)),
                    val optimizer : Optimizer = LBFGSB,
                    val n_restarts: Int = 5,
                    val noise : Double = 0d) {
      // Noise is a relaxation parameter
      val domain         = new Domain(bounds)
      val lower_bounds   = DenseVector(bounds.map(x => x._1))
      val upper_bounds   = DenseVector(bounds.map(x => x._2))
      var past_hyper     = initialization(domain,num_samples)
      var past_evals     = DenseVector((for (i <- 0 until past_hyper.rows)
                                          yield obj(past_hyper.t(::,i))):_*)

  def tune(): OptimizerResult = {
    var best_sofar = max(past_evals)
    for (i <- 0 until num_iter) {
      val next_eval = chooser(acquisition(i),
                              optimizer,
                              gpr(past_hyper,past_evals,kernel,kernelParams,noise),
                              past_evals,
                              domain,
                              lower_bounds,
                              upper_bounds,
                              n_restarts)
      val next_value = obj(next_eval)
      if (next_value >= best_sofar)
        best_sofar = next_value
      println(s"$i | $num_iter  current evaluation: $next_eval   -> score : $next_value \n best so far: $best_sofar")
      past_hyper = DenseMatrix.vertcat(past_hyper,next_eval.toDenseMatrix)
      past_evals = DenseVector.vertcat(past_evals,DenseVector[Double](next_value)) }
    val idx_argmax = argmax(past_evals)
    val result = new OptimizerResult(past_evals(idx_argmax),
                                     past_hyper.t(::,idx_argmax),
                                     past_hyper,
                                     past_evals)
    result
  }

}
