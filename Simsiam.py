from lightly.models.modules import SimSiamProjectionHead, SimSiamPredictionHead
import torch


class SimSiam(torch.nn.Module):
    def __init__(
            self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

    def forward(self, x, rev):
        out_put, out_feature = self.backbone(x, rev)
        # get projections
        f = out_feature.flatten(start_dim=1)
        z = self.projection_head(f)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()

        return z, p, out_put
