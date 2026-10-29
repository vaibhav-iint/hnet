"""Hierarchy-use interventions: residual path-drop, main-path auxiliary head, router-entropy penalty."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params
from hnet.model import build_model

CFG = "configs/models/tiny_hnet2_2m.yaml"


def _model(**dc):
    cfg = model_config_from_yaml(CFG)
    for k, v in dc.items():
        setattr(cfg.dc, k, v)
    torch.manual_seed(0)
    return build_model(cfg), cfg


def _loss(model, ids):
    out = model(ids, labels=ids)
    return out, out.loss + out.aux_loss


def test_defaults_are_off():
    base, _ = _model()
    assert base.main_aux_head is None
    assert all(getattr(st, "residual_dropout", 0.0) == 0.0 for st in base.backbone.stages())


@pytest.mark.parametrize("dc", [{"router_entropy_weight": 0.1}, {"main_aux_weight": 0.5}])
def test_aux_terms_increase_aux_loss_and_get_gradients(dc):
    ids = torch.randint(0, 256, (2, 128))
    base, _ = _model()
    m, cfg = _model(**dc)
    m.load_state_dict(base.state_dict(), strict=False)
    o0, _ = _loss(base, ids)
    o1, total = _loss(m, ids)
    torch.testing.assert_close(o1.loss, o0.loss)  # the LM loss itself is unchanged
    assert o1.aux_loss > o0.aux_loss
    total.backward()
    if "main_aux_weight" in dc:
        assert count_params(cfg) == sum(p.numel() for p in m.parameters())
        assert m.main_aux_head.weight.grad.abs().sum() > 0
        assert "main_aux_ce" in o1.stats
        # the aux head trains the hierarchy below stage 0 directly: main-network params get gradient from it alone
        m.zero_grad()
        (m(ids, labels=ids).aux_loss).backward()
        g = sum(p.grad.abs().sum() for n, p in m.named_parameters()
                if n.startswith("backbone.main_network.main_network") and p.grad is not None)
        assert g > 0


def test_residual_dropout_only_in_training_and_per_segment():
    m, _ = _model(residual_dropout=[1.0, 1.0])
    base, _ = _model()
    m.load_state_dict(base.state_dict())
    ids = torch.randint(0, 256, (2, 96))
    m.eval()
    base.eval()
    torch.testing.assert_close(m(ids, labels=ids).loss, base(ids, labels=ids).loss)  # eval: no dropout
    # with p = 1 every segment's residual is dropped in training: the model then equals one whose residual
    # projection is zero
    m.train()
    zero = _model()[0]
    zero.load_state_dict(base.state_dict())
    with torch.no_grad():
        for st in zero.backbone.stages():
            if not st.is_innermost:
                st.residual_proj.weight.zero_()
                if st.residual_proj.bias is not None:
                    st.residual_proj.bias.zero_()
    zero.train()
    torch.testing.assert_close(m(ids, labels=ids).loss, zero(ids, labels=ids).loss)


def test_residual_identity_init_survives_init_weights():
    m, _ = _model(residual_init="identity")
    m.init_weights()
    for st in m.backbone.stages():
        if not st.is_innermost:
            torch.testing.assert_close(st.residual_proj.weight, torch.eye(st.residual_proj.weight.shape[0]))
    z, _ = _model()
    z.init_weights()
    for st in z.backbone.stages():
        if not st.is_innermost:
            assert st.residual_proj.weight.abs().sum() == 0


def test_residual_warmup_scales_residual_by_progress():
    m, _ = _model(residual_warmup=0.5)
    base, _ = _model()
    m.load_state_dict(base.state_dict())
    with torch.no_grad():  # make the residual path matter (it is zero-init)
        for mm in (m, base):
            for st in mm.backbone.stages():
                if not st.is_innermost:
                    torch.nn.init.eye_(st.residual_proj.weight)
    ids = torch.randint(0, 256, (2, 96))
    m.eval()
    base.eval()
    torch.testing.assert_close(m(ids, labels=ids).loss, base(ids, labels=ids).loss)  # default progress 1.0
    m.set_progress(0.0)
    zero = _model()[0]
    zero.load_state_dict(base.state_dict())
    zero.eval()  # residual_proj stays zero-init: equals warmup at progress 0
    torch.testing.assert_close(m(ids, labels=ids).loss, zero(ids, labels=ids).loss)
    m.set_progress(0.6)
    torch.testing.assert_close(m(ids, labels=ids).loss, base(ids, labels=ids).loss)


def test_residual_warmup_scales_residual_by_progress():
    m, _ = _model(residual_warmup=0.5)
    base, _ = _model()
    m.load_state_dict(base.state_dict())
    with torch.no_grad():  # make the residual path matter (it is zero-init)
        for mm in (m, base):
            for st in mm.backbone.stages():
                if not st.is_innermost:
                    torch.nn.init.eye_(st.residual_proj.weight)
    zero = _model()[0]
    zero.load_state_dict(base.state_dict())
    with torch.no_grad():
        for st in zero.backbone.stages():
            if not st.is_innermost:
                st.residual_proj.weight.zero_()
                st.residual_proj.bias.zero_()
    ids = torch.randint(0, 256, (2, 96))
    for mm in (m, base, zero):
        mm.eval()
    torch.testing.assert_close(m(ids, labels=ids).loss, base(ids, labels=ids).loss)  # default progress 1.0
    m.set_progress(0.0)  # residual fully off
    torch.testing.assert_close(m(ids, labels=ids).loss, zero(ids, labels=ids).loss)
    m.set_progress(0.6)  # past the warmup: full residual
    torch.testing.assert_close(m(ids, labels=ids).loss, base(ids, labels=ids).loss)
