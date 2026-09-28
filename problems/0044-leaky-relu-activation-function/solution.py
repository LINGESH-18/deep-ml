def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	if z<0:
		return alpha * z
	return  z
