import breeze.linalg._
import math.ceil
import math.floor

class Model {

  def check_within_bounds(x: DenseVector[Double],bounds: DenseMatrix[Double]): Boolean = {
    val lower_bounds = bounds(::,0)
    val upper_bounds = bounds(::,1)
    x.forall((i, v) => (lower_bounds(i) <= v && v <= upper_bounds(i)))
  }

  def vector_rounded(x: DenseVector[Double]): DenseVector[Double] = {
    val rounded = for (i <- 0 until x.length)
                yield toClosestInt(x(i))
    DenseVector(rounded:_*)
  }
  def matrix_rounded(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val rounded = for (i <- 0 until x.cols)
                    yield vector_rounded(x(::,i))
    DenseMatrix(rounded:_*).t
  }

  def toClosestInt(x: Double): Double = {
    val a = if (x - floor(x) <= 0.5) floor(x) else ceil(x)
    a
  }

}
