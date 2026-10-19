import hashlib, unittest
from probe import vote_page

class VoteTests(unittest.TestCase):
    def test_complementary_corruption_recovers_genuine_gold(self):
        gold=bytes(range(256))*48
        views=[bytearray(gold) for _ in range(3)]
        for i,v in enumerate(views): v[i*4096:(i+1)*4096]=bytes([111+i])*4096
        result=b''.join(vote_page([bytes(v[p:p+4096]) for v in views])[0] for p in range(0,len(gold),4096))
        self.assertEqual(hashlib.sha256(result).digest(),hashlib.sha256(gold).digest())
    def test_zero_is_not_assumed_wrong(self):
        self.assertEqual(vote_page([b'\0'*4096,b'x'*4096,b'\0'*4096])[0],b'\0'*4096)
    def test_tie_retains_first_and_reports_unresolved(self):
        self.assertEqual(vote_page([b'a',b'b',b'c']),(b'a','tie'))
    def test_common_corruption_rejected_by_original_digest(self):
        gold=b'gold'*1024; bad=b'bad!'*1024
        candidate=vote_page([bad,bad,bad])[0]
        self.assertNotEqual(hashlib.sha256(candidate).digest(),hashlib.sha256(gold).digest())

if __name__=='__main__': unittest.main()
