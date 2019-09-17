import dk.gp.gpr._
import dk.gp.cov.CovSEiso
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import breeze.stats.distributions._
import math.Pi
import smile._
import smile.plot.{plot => plt}
import smile.clustering._
import smile.math.Math.pdist
import scala.math.floor
import smile.validation
import smile.stat.distribution.MultivariateGaussianMixture


object Main0 extends App {

  /**def ackley_(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    def ackley_vector(x: DenseVector[Double]): Double= {
      val a           = -20.0
      val b           = -0.2
      val c           = 2.0*Pi
      val d           = x.length.toDouble
      val sum_sq_term = a * exp(sqrt(sum(x:^2.0)/d) * b)
      val sum_cos_term = -1 * exp(sum(cos(x:*=c))/d)
      -a + exp(1.0) + sum_sq_term + sum_cos_term
    }
    val n_samples   = x.rows
    val ack         = for (i <- 0 until n_samples)
                      yield -ackley_vector(x.t(::,i))
    DenseMatrix(ack:_*)
  }

  def sin_(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    x :* sin(x)
  }
  def gramacy(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    (sin(10*Pi*x) :/ x:*=2.0 ) + (x:+= 1.0):^4.0
  }
  def square(x: DenseMatrix[Double]): DenseMatrix[Double]= {
    def square_vector(x: DenseVector[Double]): Double = {
      sum(x:^2.0)
    }
    val dim = x.rows
    val sq  = for (i<- 0 until dim)
              yield -square_vector(x.t(::,i))/dim.toDouble
    DenseMatrix(sq:_*)
  }
  val dim       = 2
  val n_samples = 10
  val n_iter    = 20
  val n_subiter = 15
  val bnds  = for (i <- 0 until dim)
              yield (- 10.0,10.0)
  val bounds   =  DenseMatrix(bnds:_*)
  //println(Bayesopt(ackley_,bounds,covFunc,covFuncParams,5,0,20))
  val optim     =  IBOhardBounds(ackley_,bounds,n_samples,n_subiter,power = 0.9)
  println(optim)
  //val optim     =  Bayesopt(square,bounds,n_samples,n_iter,discrete_inputs = true)
  val optim_val =  ackley_(optim)
  println(optim)
  val evaluations = IBOhardBounds.total_evaluations(n_samples,n_iter,n_subiter)
  //println(s"obj optim val  is $optim_val, total evaluations $evaluations")
  //println(s"obj optim val  is at $optim")
  //val iris = read.arff("../iris.arff", 4)
  val clus = read.table("../clustering.csv",delimiter = ";")
  val ground_truth = read.table("../groundtruth.csv",delimiter = ";")
  val x = clus.unzip
  val gt = ground_truth.unzip
  //val clusters = kmeans(x, 15, runs = 20)
  val real_labels = Metrics.getLabel(x,gt)
  def eval_cluster(i: DenseMatrix[Double]): DenseMatrix[Double] = {
    val real_labels = Metrics.getLabel(x,gt)
    def eval_dub(j: DenseVector[Double]): Double = {
      //val clust = kmeans(x,j(0).toInt,runs = 20)
      val cluster     = birch(x,j(0).toInt,j(1).toInt,j(2).toInt,20000)
      //val pred_labels = clust.getClusterLabel
      //validation.adjustedRandIndex(real_labels,pred_labels)
      if (cluster.centroids.length == 0)
        0.0
      else
        validation.adjustedRandIndex(real_labels,Metrics.getLabel(x,cluster.centroids))
    }
    val evals = for (k <- 0 until i.rows)
                  yield eval_dub(i.t(::,k))
    DenseMatrix(evals:_*)
  }
  val bounds = DenseMatrix((2.0,100.0),(2.0,100.0),(2.0,5000.0))
  val n_samples = 20
  val n_iter = 100
  val optim = Bayesopt(eval_cluster,bounds,n_samples,n_iter,discrete_inputs = true)
  println(eval_cluster(optim))
  val clusters = birch(x,5,3,5000,1000)
  println(validation.adjustedRandIndex(Metrics.getLabel(x,gt),Metrics.getLabel(x,clusters.centroids)))*/
}
