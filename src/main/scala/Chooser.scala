import breeze.linalg._
import dk.gp.gpr._

trait Chooser {
  def apply(acquisition: Acquisition,
            optimizer : Optimizer,
            gp : GprModel,
            past_evals: DenseVector[Double],
            domain: Domain,
            lower_bounds: DenseVector[Double],
            upper_bounds: DenseVector[Double],
            n_restarts: Int): DenseVector[Double]
}

object MaxAcquisition extends Chooser {
  def apply(acquisition: Acquisition,
            optimizer : Optimizer,
            gp : GprModel,
            past_evals: DenseVector[Double],
            domain : Domain,
            lower_bounds: DenseVector[Double],
            upper_bounds: DenseVector[Double],
            n_restarts: Int) : DenseVector[Double] = {
      val list_minimums = (0 until n_restarts).map(j => optimizer(acquisition,gp,past_evals,domain,lower_bounds,upper_bounds) )
      val sorted_mins = list_minimums.sortWith((x,y) => -acquisition.eval(x,gp,past_evals) <= -acquisition.eval(x,gp,past_evals))
      sorted_mins(0)
            }
}
