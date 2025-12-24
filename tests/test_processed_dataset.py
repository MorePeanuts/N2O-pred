from n2o_pred.data import SequentialN2ODataset


class TestRawSequentialDataset:
    dataset = SequentialN2ODataset()

    def test_sowdurs_uniqueness(self):
        for seq_data in self.dataset:
            assert len(set(seq_data.sowdurs)) == len(seq_data.sowdurs), (
                f'seq_id: {seq_data.seq_id}, sowdurs: {seq_data.sowdurs}'
            )

    def test_no_of_obs_uniqueness(self):
        for seq_data in self.dataset:
            assert len(set(seq_data.no_of_obs)) == len(seq_data.no_of_obs), (
                f'seq_id: {seq_data.seq_id}, no_of_obs: {seq_data.no_of_obs}'
            )


class TestExpandSequentialDataset:
    pass
