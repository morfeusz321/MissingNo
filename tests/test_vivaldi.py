import unittest
from datetime import timedelta
from vivaldi.vivaldi_coordinates import NetworkNode


class TestNetworkCoordinate(unittest.TestCase):
    
    def assertApproxEqual(self, a, b, tol=1.0):
        self.assertAlmostEqual(a, b, delta=tol)

    def test_convergence(self):
        a = NetworkNode(2)
        b = NetworkNode(3)
        t = timedelta(milliseconds=250)
        for _ in range(20):
            a.update(b, t)
            b.update(a, t)
        rtt = a.estimated_rtt(b).total_seconds() * 1000.0
        self.assertApproxEqual(rtt, 250.0, tol=1.0)

    def test_mini_network(self):
        slc = NetworkNode(2)
        nyc = NetworkNode(2)
        lax = NetworkNode(2)
        mad = NetworkNode(2)
        
        error = (slc.error + nyc.error + lax.error + mad.error) ** 0.5
        self.assertApproxEqual(error, 20.0, tol=20.0)

        # iterate to converge
        for _ in range(20):
            slc.update(nyc, timedelta(milliseconds=162))
            nyc.update(slc, timedelta(milliseconds=162))
            slc.update(lax, timedelta(milliseconds=115))
            lax.update(slc, timedelta(milliseconds=115))
            slc.update(mad, timedelta(milliseconds=242))
            mad.update(slc, timedelta(milliseconds=242))
            nyc.update(lax, timedelta(milliseconds=95))
            lax.update(nyc, timedelta(milliseconds=95))
            nyc.update(mad, timedelta(milliseconds=168))
            mad.update(nyc, timedelta(milliseconds=168))
            lax.update(mad, timedelta(milliseconds=192))
            mad.update(lax, timedelta(milliseconds=192))

        error = slc.error + nyc.error + lax.error + mad.error
        self.assertLess(error, 5.0)

    # def test_serde(self):
    #     s = '{"coordinates":[1.5,0.5,2.0],"height":0.1,"error":1.0}'
    #     a = NetworkCoordinate3D.from_json(s)
    #     self.assertApproxEqual(a.heightvec.len(), 2.649509, tol=0.001)
    #     self.assertApproxEqual(a.error(), 1.0, tol=1.0)
    #     self.assertEqual(a.estimated_rtt(a).total_seconds() * 1000, 0)
    #     t = a.to_json()
    #     self.assertEqual(t, s)
    #
    # def test_estimated_rtt(self):
    #     s = '{"coordinates":[1.5,0.5,2.0],"height":25.0,"error":1.0}'
    #     a = NetworkCoordinate3D.from_json(s)
    #     s = '{"coordinates":[-1.5,-0.5,-2.0],"height":50.0,"error":1.0}'
    #     b = NetworkCoordinate3D.from_json(s)
    #     estimate = a.estimated_rtt(b).total_seconds()
    #     self.assertApproxEqual(estimate, 0.080099, tol=0.001)
    #
    # def test_error_getter(self):
    #     s = '{"coordinates":[1.5,0.5,2.0],"height":25.0,"error":1.0}'
    #     a = NetworkCoordinate3D.from_json(s)
    #     self.assertApproxEqual(a.error(), 1.0, tol=1.0)


if __name__ == "__main__":
    unittest.main()
