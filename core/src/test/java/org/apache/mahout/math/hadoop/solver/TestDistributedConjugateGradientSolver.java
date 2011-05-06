package org.apache.mahout.math.hadoop.solver;

import java.io.File;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DiagonalOffsetLinearOperator;
import org.apache.mahout.math.LinearOperator;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.SquaredLinearOperator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;
import org.junit.Test;


public class TestDistributedConjugateGradientSolver extends MahoutTestCase
{
  private Vector randomVector(int size, double entryMean) {
    DenseVector v = new DenseVector(size);
    Random r = new Random(1234L);
    
    for (int i = 0; i < size; ++i) {
      v.setQuick(i, r.nextGaussian() * entryMean);
    }
    
    return v;
  }

  @Test
  public void testSolver() throws Exception {
    File testData = getTestTempDir("testdata");
    DistributedRowMatrix matrix = new TestDistributedRowMatrix().randomDistributedMatrix(
        10, 10, 10, 10, 10.0, true, testData.getAbsolutePath());
    matrix.setConf(new Configuration());
    Vector vector = randomVector(matrix.numCols(), 10.0);
    
    DistributedConjugateGradientSolver solver = new DistributedConjugateGradientSolver();
    Vector x = solver.solve(matrix, vector);

    Vector solvedVector = matrix.times(x);    
    double distance = Math.sqrt(vector.getDistanceSquared(solvedVector));
    assertEquals(0.0, distance, EPSILON);
  }

  @Test
  public void testSolverAsymmetric() throws Exception {
    File testData = getTestTempDir("testdata");
    DistributedRowMatrix matrix = new TestDistributedRowMatrix().randomDistributedMatrix(
        50, 20, 10, 10, 10.0, false, testData.getAbsolutePath());
    matrix.setConf(new Configuration());
    Vector vector = randomVector(matrix.numCols(), 10.0);
    
    LinearOperator squaredMatrix = new SquaredLinearOperator(matrix);
    
    DistributedConjugateGradientSolver solver = new DistributedConjugateGradientSolver();
    Vector x = solver.solve(squaredMatrix, vector);

    Vector solvedVector = squaredMatrix.times(x);
    double distance = Math.sqrt(vector.getDistanceSquared(solvedVector));
    assertEquals(0.0, distance, EPSILON);
  }
  
  @Test
  public void testSolverLambda() throws Exception {
    File testData = getTestTempDir("testdata");
    DistributedRowMatrix matrix = new TestDistributedRowMatrix().randomDistributedMatrix(
        50, 5, 10, 5, 10.0, false, testData.getAbsolutePath());
    matrix.setConf(new Configuration());
    Vector vector = randomVector(matrix.numCols(), 10.0);
    
    LinearOperator offsetMatrix = new DiagonalOffsetLinearOperator(new SquaredLinearOperator(matrix), 0.1);
    
    DistributedConjugateGradientSolver solver = new DistributedConjugateGradientSolver();
    Vector x = solver.solve(offsetMatrix, vector);

    Vector solvedVector = offsetMatrix.times(x);
    
    double distance = Math.sqrt(vector.getDistanceSquared(solvedVector));
    assertEquals(0.0, distance, EPSILON);
  }  
}
