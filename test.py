def Verify(self, msg, sig_bytes):
	mac = self.Sign(msg)
	return self.Sign(mac) == self.Sign(sig_bytes)
