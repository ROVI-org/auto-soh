import numpy as np

from moirae.estimators.online.filters.conversions import (IdentityConversionOperator,
                                                          LinearConversionOperator,
                                                          AbsoluteValueConversionOperator)


def test_identity_operator():
    # Setup samples and covariance
    single_sample = np.array([0, 1, 2])  # single sample from 3D space (flattened)
    multiple_samples = np.arange(15).reshape((5, 3))  # 5 samples from 3D space
    covariance = np.diag([1, 2, 3])

    # Conversion
    convertor = IdentityConversionOperator()
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(single_sample, transform_single_sample)
    assert np.allclose(multiple_samples, transform_multi_samples)
    assert np.allclose(covariance, transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)


def test_linear_conversions():
    # Setup samples and covariance
    single_sample = np.array([0, 1, 2])  # single sample from 3D space (flattened)
    multiple_samples = np.arange(15).reshape((5, 3))  # 5 samples from 3D space
    covariance = np.diag([1, 2, 3])

    # Base conversion
    convertor = LinearConversionOperator()
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(single_sample, transform_single_sample)
    assert np.allclose(multiple_samples, transform_multi_samples)
    assert np.allclose(covariance, transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)

    # Simple bias (+ simple multi)
    bias = np.array(10)
    convertor = LinearConversionOperator(additive_array=bias)
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(single_sample + bias, transform_single_sample)
    assert np.allclose(multiple_samples + bias, transform_multi_samples)
    assert np.allclose(covariance, transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)

    # 1D bias (+ simple multi)
    bias = np.array([10, 11, 12])
    convertor = LinearConversionOperator(additive_array=bias)
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(single_sample + bias, transform_single_sample)
    assert np.allclose(multiple_samples + bias, transform_multi_samples)
    assert np.allclose(covariance, transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)

    # Simple multi (+ simple bias)
    multi = np.array(5)
    convertor = LinearConversionOperator(multiplicative_array=multi)
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(multi * single_sample, transform_single_sample)
    assert np.allclose(multi * multiple_samples, transform_multi_samples)
    assert np.allclose(multi * multi * covariance, transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)

    # 1D multi (+ simple bias)
    multi = np.array([5, 6, 7])
    convertor = LinearConversionOperator(multiplicative_array=multi)
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(multi * single_sample, transform_single_sample)
    assert np.allclose(multi * multiple_samples, transform_multi_samples)
    assert np.allclose(multi * multi * covariance, transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)

    # 2D full-rank multi (+ simple bias)
    multi = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
    convertor = LinearConversionOperator(multiplicative_array=multi)
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(np.matmul(single_sample, multi), transform_single_sample)
    assert np.allclose(np.matmul(multiple_samples, multi), transform_multi_samples)
    assert np.allclose(np.matmul(np.matmul(multi.T, covariance), multi), transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)

    # Simple multi simple bias
    bias = np.array(10)
    multi = np.array(5)
    convertor = LinearConversionOperator(multiplicative_array=multi, additive_array=bias)
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(multi * single_sample + bias, transform_single_sample)
    assert np.allclose(multi * multiple_samples + bias, transform_multi_samples)
    assert np.allclose(multi * multi * covariance, transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)

    # 1D bias + 2D multi
    bias = np.array([10, 11, 12])
    multi = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]])
    convertor = LinearConversionOperator(multiplicative_array=multi, additive_array=bias)
    # transform
    transform_single_sample = convertor.transform_samples(samples=single_sample)
    transform_multi_samples = convertor.transform_samples(samples=multiple_samples)
    transform_cov = convertor.transform_covariance(covariance=covariance)
    # invert
    reconverted_single_sample = convertor.inverse_transform_samples(transformed_samples=transform_single_sample)
    reconverted_multi_samples = convertor.inverse_transform_samples(transformed_samples=transform_multi_samples)
    reconverted_cov = convertor.inverse_transform_covariance(transform_cov)
    # Checks
    assert np.allclose(np.matmul(single_sample, multi) + bias, transform_single_sample)
    assert np.allclose(np.matmul(multiple_samples, multi) + bias, transform_multi_samples)
    assert np.allclose(np.matmul(np.matmul(multi.T, covariance), multi), transform_cov)
    assert np.allclose(single_sample, reconverted_single_sample)
    assert np.allclose(multiple_samples, reconverted_multi_samples)
    assert np.allclose(covariance, reconverted_cov)


def test_absolute_val_conversion():
    # Initialize convertor
    convertor = AbsoluteValueConversionOperator()

    # For now, let's assume it works in 3D
    samples = np.hstack((-np.arange(1, 11).reshape((-1, 1)),
                         -np.arange(20, 10, -1).reshape(-1, 1),
                         np.zeros((10, 1)),
                         np.arange(1, 11).reshape((-1, 1)),
                         np.arange(20, 10, -1).reshape((-1, 1))))
    mean = np.average(samples, axis=0)
    covariance = np.matmul((samples - mean).T, samples - mean)

    # convert samples
    abs_samples = convertor.transform_samples(samples=samples)
    assert np.allclose(abs_samples, abs(samples))
    # revert samples, should not change them
    invert_samples = convertor.inverse_transform_samples(transformed_samples=abs_samples)
    assert np.allclose(invert_samples, abs_samples)
    # convert covariance
    transformed_covariance = convertor.transform_covariance(covariance=covariance,
                                                            pivot=mean)
    # Calculate covariance "by hand" from the transformed samples
    abs_mean = np.average(abs_samples, axis=0)
    calc_cov = np.matmul((abs_samples - abs_mean).T, (abs_samples - abs_mean))
    assert np.allclose(transformed_covariance, calc_cov)
    # revert covariance, should not change anything regardless of pivot used
    revert_covariance = convertor.inverse_transform_covariance(transformed_covariance=transformed_covariance,
                                                               transformed_pivot=abs_mean)
    assert np.allclose(revert_covariance, transformed_covariance)
