or ind, w in enumerate(self.decode_w):
                if ind != self._last:
                z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
                            # last layer or decoder should not apply non linearities
                            kind=self._nl_type)
                z = self.decode_bn[ind](z)
                else:
                z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
                            # last layer or decoder should not apply non linearities
         