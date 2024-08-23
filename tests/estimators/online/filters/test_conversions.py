import numpy as np

from moirae.estimators.online.filters.conversions import LinearConversionOperator


def test_conversions():
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
