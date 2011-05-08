package org.apache.mahout.math;

import org.junit.Test;

public class TestLinearOperators extends MahoutTestCase {
  private double[][] aValues = 
    {{-0.627806977546231, 0.484226464067699,  0.399781846486140},
     {-0.723024724663656, 0.188554635827914, -0.371346469977094},     
     {-1.096603951039857, 0.134997337811092, -1.227047004525883}};

  private double[][] bValues =
    {{-2.1542444851275238, -0.2469499595545392,  0.0537495438547667},
     {-0.6642610710366451,  0.2340196616232284,  1.8846451212749238},
     {-0.2500161918313162, -1.1493655781396637, -0.6430981471031478}};
  
  private DenseMatrix a = new DenseMatrix(aValues);
  private DenseMatrix b = new DenseMatrix(bValues);
  
  private class WrappedLinearOperator extends AbstractLinearOperator {
    private Matrix matrix;

    public WrappedLinearOperator(Matrix matrix) {
      this.matrix = matrix;
    }
    
    @Override
    public LinearOperator transpose() {
      return new WrappedLinearOperator(matrix.transpose());
    }

    @Override
    public int numRows() {
      return matrix.numRows();
    }

    @Override
    public int numCols() {
      return matrix.numCols();
    }

    @Override
    public Vector times(Vector v) {
      return matrix.times(v);
    }
  }
  
  LinearOperator linopA = new WrappedLinearOperator(a);
  LinearOperator linopB = new WrappedLinearOperator(b);
  
  private DenseVector v = new DenseVector(new double[] {0.8134542290782942, 1.6289304763813024, -0.0733365425049696});
  private DenseVector av = new DenseVector(new double[] { 0.248760385482841, -0.253771861063209, -0.582148459003500});
  private DenseVector aav = new DenseVector(new double[] { 0.66569612446439475, -0.00598199115997356, 0.90801069383587796 });
  
  private Vector w = new DenseVector(new double[] {-1.235246340263785, -0.769178028745684, 0.648424632385750});
  private Vector ones = new DenseVector(new double[] {1.0, 1.0, 1.0});

  protected static void assertEquals(Vector expected, Vector result) {
    double distance = Math.sqrt(expected.getDistanceSquared(result));
    assertEquals(0.0, distance, EPSILON);
  }
  
  @Test
  public void testApplication() {
    assertEquals(av, a.times(v));    
    assertEquals(aav, a.timesSquared(v));

    assertEquals(av, linopA.times(v));
    assertEquals(aav, linopA.timesSquared(v));
  }
  
  @Test
  public void testSquaredLinearOperator() {
    LinearOperator ata = new SquaredLinearOperator(a);
    assertEquals(aav, ata.times(v));
    assertEquals(aav, ata.transpose().times(v));
  }
  
  @Test
  public void testLinearOperatorSum() {
    LinearOperator sum = a.plus(b);
    
    // the matrix implementation will sum the matrices instead of constructing a SumLinearOperator
    assertTrue(sum instanceof Matrix);
    
    // the wrapped linops will use the generic implementation
    LinearOperator sumLinop = linopA.plus(linopB);
    assertFalse(sumLinop instanceof Matrix);
    
    Vector expected = a.times(v).plus(b.times(v));
    
    assertEquals(expected, sum.times(v));
    assertEquals(expected, sumLinop.times(v));
  }
  
  @Test
  public void testLinearOperatorProduct() {
    LinearOperator product = a.times(b);
    
    // the matrix implementation will sum the matrices instead of constructing a SumLinearOperator
    assertTrue(product instanceof Matrix);
    
    // the wrapped linops will use the generic implementation
    LinearOperator productLinop = linopA.times(linopB);
    assertFalse(productLinop instanceof Matrix);
    
    Vector expected = a.times(b.times(v));
    
    assertEquals(expected, product.times(v));
    assertEquals(expected, productLinop.times(v));
  }
  
  @Test
  public void testLinearOperatorScaling() {
    LinearOperator scaled = a.times(2.0);
    assertTrue(scaled instanceof Matrix);
    
    LinearOperator scaledLinop = linopA.times(2.0);
    assertFalse(scaledLinop instanceof Matrix);
    
    Vector expected = a.times(2.0).times(v);
    
    assertEquals(expected, scaled.times(v));
    assertEquals(expected, scaledLinop.times(v));
  }
  
  @Test
  public void testLinearOperatorTranspose() {
    assertEquals(a.transpose().times(v), linopA.transpose().times(v));
    assertEquals(b.transpose().times(a.transpose().times(v)), linopA.times(linopB).transpose().times(v));
    assertEquals(a.transpose().plus(b.transpose()).times(v), linopA.plus(linopB).transpose().times(v));
    assertEquals(a.transpose().times(0.1).times(v), linopA.times(0.1).transpose().times(v));    
  }
  
  @Test
  public void testDiagonalOffsetLinearOperator() {
    LinearOperator offset1 = new DiagonalOffsetLinearOperator(a, 0.1);
    LinearOperator offset2 = new DiagonalOffsetLinearOperator(a, w);

    assertEquals(a.times(v).plus(v.times(0.1)), offset1.times(v));  
    assertEquals(a.times(v).plus(v.times(w)), offset2.times(v));
    
    assertEquals(a.transpose().times(v).plus(v.times(0.1)), offset1.transpose().times(v));
    assertEquals(a.transpose().times(v).plus(v.times(w)), offset2.transpose().times(v));
  }
  
  @Test
  public void testRowOffsetLinearOperator() {    
    LinearOperator offset = new RowOffsetLinearOperator(a, w);
    
    assertEquals(a.times(v).plus(ones.times(w.dot(v))), offset.times(v));
    assertEquals(a.transpose().times(v).plus(w.times(ones.dot(v))), offset.transpose().times(v));
    
    offset = new RowOffsetLinearOperator(a, 0.1);

    assertEquals(a.times(v).plus(ones.times(ones.times(0.1).dot(v))), offset.times(v));
    assertEquals(a.transpose().times(v).plus(ones.times(0.1).times(ones.dot(v))), offset.transpose().times(v));
    
    // for scalar offsets, both row and column offset operators should give the same results
    
    LinearOperator colOffset = new ColumnOffsetLinearOperator(a, 0.1);
    assertEquals(offset.times(v), colOffset.times(v));
  }
  
  @Test
  public void testColumnOffsetLinearOperator() {
    LinearOperator offset = new ColumnOffsetLinearOperator(a, w);
    assertEquals(a.times(v).plus(w.times(ones.dot(v))), offset.times(v));
    assertEquals(a.transpose().times(v).plus(ones.times(w.dot(v))), offset.transpose().times(v));
    
    offset = new ColumnOffsetLinearOperator(a, 0.1);
    assertEquals(a.times(v).plus(ones.times(0.1).times(ones.dot(v))), offset.times(v));
    assertEquals(a.transpose().times(v).plus(ones.times(ones.times(0.1).dot(v))), offset.transpose().times(v));
  }
}
