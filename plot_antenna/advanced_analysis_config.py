"""Typed configuration groups for the advanced-analysis plot dispatcher (#37).

``generate_advanced_analysis_plots`` historically took 30+ flat keyword
arguments grouped only by comment banners (link-budget / indoor / fading / MIMO /
wearable). These frozen dataclasses give each group a name, type, and the same
defaults, so the dispatcher body and any new callers operate on cohesive objects
instead of a flat bag. The public flat signature is kept for back-compat and is
bundled into these via :func:`configs_from_flat_kwargs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class LinkBudgetConfig:
    enabled: bool = False
    pt_dbm: float = 0.0
    pr_dbm: float = -98.0
    gr_dbi: float = 0.0
    path_loss_exp: float = 2.0
    misc_loss_db: float = 10.0
    target_range_m: float = 5.0


@dataclass(frozen=True)
class IndoorConfig:
    enabled: bool = False
    environment: str = "Office"
    path_loss_exp: float = 3.0
    n_walls: int = 1
    wall_material: str = "drywall"
    shadow_fading_db: float = 5.0
    max_distance_m: float = 30.0


@dataclass(frozen=True)
class FadingConfig:
    enabled: bool = False
    pr_sensitivity_dbm: float = -98.0
    pt_dbm: float = 0.0
    target_reliability: float = 99.0
    model: str = "rayleigh"
    rician_k: float = 10.0
    realizations: int = 1000


@dataclass(frozen=True)
class MimoConfig:
    enabled: bool = False
    snr_db: float = 20.0
    fading_model: str = "rayleigh"
    rician_k: float = 10.0
    xpr_db: float = 6.0
    ecc_values: Optional[List[float]] = None
    gain_data_list: Optional[List] = None


@dataclass(frozen=True)
class WearableConfig:
    enabled: bool = False
    body_positions: Optional[List] = None
    tx_power_mw: float = 1.0
    num_devices: int = 20
    room_size: Tuple[float, float, float] = (10, 10, 3)


@dataclass(frozen=True)
class AdvancedAnalysisConfig:
    """Bundle of all five advanced-analysis configuration groups."""

    link_budget: LinkBudgetConfig = LinkBudgetConfig()
    indoor: IndoorConfig = IndoorConfig()
    fading: FadingConfig = FadingConfig()
    mimo: MimoConfig = MimoConfig()
    wearable: WearableConfig = WearableConfig()


def configs_from_flat_kwargs(
    *,
    link_budget_enabled: bool = False,
    lb_pt_dbm: float = 0.0,
    lb_pr_dbm: float = -98.0,
    lb_gr_dbi: float = 0.0,
    lb_path_loss_exp: float = 2.0,
    lb_misc_loss_db: float = 10.0,
    lb_target_range_m: float = 5.0,
    indoor_enabled: bool = False,
    indoor_environment: str = "Office",
    indoor_path_loss_exp: float = 3.0,
    indoor_n_walls: int = 1,
    indoor_wall_material: str = "drywall",
    indoor_shadow_fading_db: float = 5.0,
    indoor_max_distance_m: float = 30.0,
    fading_enabled: bool = False,
    fading_pr_sensitivity_dbm: float = -98.0,
    fading_pt_dbm: float = 0.0,
    fading_target_reliability: float = 99.0,
    fading_model: str = "rayleigh",
    fading_rician_k: float = 10.0,
    fading_realizations: int = 1000,
    mimo_enabled: bool = False,
    mimo_snr_db: float = 20.0,
    mimo_fading_model: str = "rayleigh",
    mimo_rician_k: float = 10.0,
    mimo_xpr_db: float = 6.0,
    mimo_ecc_values: Optional[List[float]] = None,
    mimo_gain_data_list: Optional[List] = None,
    wearable_enabled: bool = False,
    wearable_body_positions: Optional[List] = None,
    wearable_tx_power_mw: float = 1.0,
    wearable_num_devices: int = 20,
    wearable_room_size: Tuple[float, float, float] = (10, 10, 3),
) -> AdvancedAnalysisConfig:
    """Bundle the historical flat keyword arguments into grouped configs."""
    return AdvancedAnalysisConfig(
        link_budget=LinkBudgetConfig(
            enabled=link_budget_enabled,
            pt_dbm=lb_pt_dbm,
            pr_dbm=lb_pr_dbm,
            gr_dbi=lb_gr_dbi,
            path_loss_exp=lb_path_loss_exp,
            misc_loss_db=lb_misc_loss_db,
            target_range_m=lb_target_range_m,
        ),
        indoor=IndoorConfig(
            enabled=indoor_enabled,
            environment=indoor_environment,
            path_loss_exp=indoor_path_loss_exp,
            n_walls=indoor_n_walls,
            wall_material=indoor_wall_material,
            shadow_fading_db=indoor_shadow_fading_db,
            max_distance_m=indoor_max_distance_m,
        ),
        fading=FadingConfig(
            enabled=fading_enabled,
            pr_sensitivity_dbm=fading_pr_sensitivity_dbm,
            pt_dbm=fading_pt_dbm,
            target_reliability=fading_target_reliability,
            model=fading_model,
            rician_k=fading_rician_k,
            realizations=fading_realizations,
        ),
        mimo=MimoConfig(
            enabled=mimo_enabled,
            snr_db=mimo_snr_db,
            fading_model=mimo_fading_model,
            rician_k=mimo_rician_k,
            xpr_db=mimo_xpr_db,
            ecc_values=mimo_ecc_values,
            gain_data_list=mimo_gain_data_list,
        ),
        wearable=WearableConfig(
            enabled=wearable_enabled,
            body_positions=wearable_body_positions,
            tx_power_mw=wearable_tx_power_mw,
            num_devices=wearable_num_devices,
            room_size=wearable_room_size,
        ),
    )
