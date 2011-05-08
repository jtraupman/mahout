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
  
  LinearOperator linopA = new AbstractLinearOperator() {
    @Override
    public int numRows() {
      return a.numRows();
    }

    @Override
    public int numCols() {
      return a.numCols();
    }

    @Override
    public Vector times(Vector v) {
      return a.times(v);
    }
  };
  
  LinearOperator linopB = new AbstractLinearOperator() {
    @Override
    public int numRows() {
      return b.numRows();
    }

    @Override
    public int numCols() {
      return b.numCols();
    }

    @Override
    public Vector times(Vector v) {
      return b.times(v);
    }
  };
  
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
    Vector result = a.times(v);
    assertEquals(av, result);
    
    result = a.timesSquared(v);
    assertEquals(aav, result);
  }
  
  @Test
  public void testSquaredLinearOperator() {
    LinearOperator ata = new SquaredLinearOperator(a);
    Vector result = ata.times(v);
    assertEquals(aav, result);
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
    Vector result1 = sum.times(v);
    Vector result2 = sumLinop.times(v);
    
    assertEquals(expected, result1);
    assertEquals(expected, result2);
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
    Vector result1 = product.times(v);
    Vector result2 = productLinop.times(v);
    
    assertEquals(expected, result1);
    assertEquals(expected, result2);
  }
  
  @Test
  public void testLinearOperatorScaling() {
    LinearOperator scaled = a.scale(2.0);
    assertTrue(scaled instanceof Matrix);
    
    LinearOperator scaledLinop = linopA.scale(2.0);
    assertFalse(scaledLinop instanceof Matrix);
    
    Vector expected = a.times(2.0).times(v);
    Vector result1 = scaled.times(v);
    Vector result2 = scaledLinop.times(v);
    
    assertEquals(expected, result1);
    assertEquals(expected, result2);
  }
  
  @Test
  public void testDiagonalOffsetLinearOperator() {
    LinearOperator offset1 = new DiagonalOffsetLinearOperator(a, 0.1);
    Vector result = offset1.times(v);
    Vector expected = a.times(v).plus(v.times(0.1));
    assertEquals(expected, result);
    
    LinearOperator offset2 = new DiagonalOffsetLinearOperator(a, w);
    result = offset2.times(v);
    expected = a.times(v).plus(v.times(w));
    assertEquals(expected, result);
  }
  
  @Test
  public void testRowOffsetLinearOperator() {    
    LinearOperator offset = new RowOffsetLinearOperator(a, w);
    
    Vector result = offset.times(v);
    Vector expected = a.times(v).plus(ones.times(w.dot(v)));
    assertEquals(expected, result);
    
    offset = new RowOffsetLinearOperator(a, 0.1);
    result = offset.times(v);
    expected = a.times(v).plus(ones.times(ones.times(0.1).dot(v)));

    assertEquals(expected, result);
    
    // for scalar offsets, both row and column offset operators should give the same results
    
    LinearOperator colOffset = new ColumnOffsetLinearOperator(a, 0.1);
    Vector colResult = colOffset.times(v);
    assertEquals(result, colResult);
  }
  
  @Test
  public void testColumnOffsetLinearOperator() {
    LinearOperator offset = new ColumnOffsetLinearOperator(a, w);
    Vector result = offset.times(v);
    Vector expected = a.times(v).plus(w.times(ones.dot(v)));
    assertEquals(expected, result);
    
    offset = new ColumnOffsetLinearOperator(a, 0.1);
    result = offset.times(v);
    expected = a.times(v).plus(ones.times(0.1).times(ones.dot(v)));
    assertEquals(expected, result);
  }
}
