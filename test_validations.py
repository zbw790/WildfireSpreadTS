#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physics Validation Experiment - One-Click Fixed Version
======================================================

核心特性
- 自动读取 test_simulation.py 的成熟组件（FixedFireSpreadSimulator / FixedFireEventLoader / SimulationConfig）
- 自动发现 fire_event（优先 fire_24461899.hdf5），找不到可生成 demo
- 对 "已选特征"（BEST_FEATURES 映射到的 FEATURE_NAMES）**逐个**做受控扰动（默认：按变量预设范围）
- 统一跑基线与扰动仿真，输出：
  * 每变量文件夹下：
      - {var}_comparison.gif          （Baseline vs Perturbed vs Actual 的 2x2 动态对比，直观展示差异）
      - {var}_fire_evolution.png      （面积随时间对比）
      - {var}_response_curve.png      （响应曲线与每日面积）
      - {var}_data.json               （该变量的完整结果数据）
  * 根目录：
      - physics_validation_results.xlsx
      - summary_statistics.json
      - physics_validation_report.txt
      - overview_dashboard.png

修复点
- 不再依赖 self.experiment；所有依赖显式注入
- ground truth 判空采用 is not None，避免真值歧义
- GIF 使用已有预测结果（baseline / perturbed），不再重复“干净复跑”
- GIF 速度可通过 --interval / --fps 一键调整
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from scipy import stats
from tqdm import tqdm

# --------- 引入成熟稳定实现（来自同目录 test_simulation.py） ----------
try:
    from test_simulation import (
        SimulationConfig, FixedFireEventLoader, FixedFireSpreadSimulator,
        load_model_with_compatibility, OfficialFireUNet, WildFireConfig, FirePredictionConfig
    )
except ImportError:
    print("Error: Cannot import from test_simulation.py; please put this file in the same folder.")
    sys.exit(1)

# 为兼容旧模型 state_dict 中的类路径
sys.modules[__name__].WildFireConfig = WildFireConfig
sys.modules[__name__].FirePredictionConfig = FirePredictionConfig


# =============================== 工具函数 ===============================
def _safe_corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    sx, sy = np.nanstd(x), np.nanstd(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = np.corrcoef(x, y)
    if np.any(np.isnan(c)):
        return 0.0
    return float(c[0, 1])


def _safe_linregress(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.nanstd(x) < 1e-12:
        return 0.0, 0.0, 0.0, 1.0, 0.0
    try:
        res = stats.linregress(x, y)
        vals = [res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr]
        if any(np.isnan(v) for v in vals):
            return 0.0, 0.0, 0.0, 1.0, 0.0
        return float(res.slope), float(res.intercept), float(res.rvalue), float(res.pvalue), float(res.stderr)
    except Exception:
        return 0.0, 0.0, 0.0, 1.0, 0.0


def _binarize(arr: np.ndarray, thr: float) -> np.ndarray:
    return (arr > thr).astype(np.float32)


def _area_px(arr: np.ndarray, thr: float) -> int:
    return int((arr > thr).sum())


def _iou(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    A = _binarize(a, thr)
    B = _binarize(b, thr)
    inter = float((A * B).sum())
    union = float(A.sum() + B.sum() - inter)
    return inter / union if union > 0 else 0.0


def _pad_hold_last(seq, L):
    out = list(seq)
    while len(out) < L:
        out.append(out[-1])
    return out[:L]


# =============================== 配置（物理假设 + 范围） ===============================
class PhysicsValidationConfig:
    def __init__(self):
        self.base = SimulationConfig()
        self.output_dir = "physics_validation"
        self.fire_threshold = 0.3
        self.sim_days_cap = 20
        # 变量 → 预期方向/物理原理/扰动范围与步数（可按需改）
        self.hypotheses = {
            'Wind_Speed': ('positive', 'Wind enhances oxygen supply and flame tilt', (-0.5, 1.0), 8),
            'Max_Temp_K': ('positive', 'Higher temp lowers ignition threshold', (-0.3, 0.5), 7),
            'Min_Temp_K': ('positive', 'Higher min temp implies drier conditions', (-0.3, 0.5), 7),
            'Slope': ('positive', 'Upslope preheats fuel ahead', (-0.4, 0.8), 7),
            'Aspect': ('variable', 'South-facing slopes receive more radiation', (-0.3, 0.3), 6),
            'Elevation': ('negative', 'Higher elevation tends to be cooler', (-0.2, 0.4), 6),
            'Landcover': ('variable', 'Different vegetation has different flammability', (-0.3, 0.5), 6),
            'NDVI': ('positive', 'Higher biomass provides more fuel', (-0.4, 0.6), 6),
            'EVI2': ('positive', 'Vegetation vigor proxy', (-0.4, 0.6), 6),
            'Total_Precip': ('negative', 'Rain increases fuel moisture', (-0.8, 1.5), 8),
            'VIIRS_M11': ('positive', 'TIR relates to surface temperature', (-0.3, 0.7), 6),
            'VIIRS_I2': ('positive', 'NIR sensitive to dry vegetation', (-0.3, 0.7), 6),
            'VIIRS_I1': ('positive', 'Red band reflects vegetation stress', (-0.3, 0.7), 6),
        }
        # 从 BEST_FEATURES → FEATURE_NAMES 建立已选变量映射
        self.feature_mapping = {}
        for i, original_idx in enumerate(self.base.BEST_FEATURES):
            if original_idx < len(self.base.FEATURE_NAMES):
                name = self.base.FEATURE_NAMES[original_idx]
                self.feature_mapping[name] = i

    def testable_variables(self):
        return [v for v in self.feature_mapping.keys() if v in self.hypotheses]


# =============================== 受控扰动实验 ===============================
class ControlledPerturbationExperiment:
    def __init__(self, simulator, loader, cfg: PhysicsValidationConfig):
        self.sim = simulator
        self.loader = loader
        self.cfg = cfg
        self.stats = loader.feature_stats  # mean/std（训练期间）

    def _apply_factor(self, tensor: torch.Tensor, feat_idx: int, factor: float):
        if tensor is None:
            return None
        out = tensor.clone()
        original_idx = self.cfg.base.BEST_FEATURES[feat_idx]
        mean = float(self.stats['mean'][original_idx]) if original_idx < len(self.stats['mean']) else 0.0
        std = float(self.stats['std'][original_idx]) if original_idx < len(self.stats['std']) else 1.0
        eps = 1e-6
        for t in range(out.shape[0]):
            norm = out[t, feat_idx]
            phys = norm * std + mean
            pert = phys * (1.0 + float(factor))
            renorm = (pert - mean) / (std + eps)
            out[t, feat_idx] = renorm
        return out

    @torch.no_grad()
    def _simulate(self, init_seq: torch.Tensor, weather: torch.Tensor, days: int):
        seq_len = self.cfg.base.SEQUENCE_LENGTH
        idx_fire = len(self.cfg.base.BEST_FEATURES) - 1
        cur = init_seq.clone()
        preds = []
        for d in range(days):
            pred = self.sim.predict_single_step(cur.unsqueeze(0))
            if not isinstance(pred, torch.Tensor):
                pred = torch.as_tensor(pred)
            while pred.dim() > 2:
                pred = pred.squeeze(0)
            pred = pred.to(torch.float32).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
            if d < days - 1:
                if weather is not None and d + 1 < (len(weather) - seq_len + 1):
                    nxt = weather[d+1: d+1+seq_len].clone()
                    nxt[-1, idx_fire] = torch.from_numpy(pred)
                    cur = nxt
                else:
                    nf = cur[-1].clone()
                    nf[idx_fire] = torch.from_numpy(pred)
                    cur = torch.cat([cur[1:], nf.unsqueeze(0)], dim=0)
        return preds

    def run_on_variable(self, event_data: torch.Tensor, var_name: str, start_day=0):
        if var_name not in self.cfg.hypotheses or var_name not in self.cfg.feature_mapping:
            raise ValueError(f"Variable {var_name} not available")

        feat_idx = self.cfg.feature_mapping[var_name]
        seq_len = self.cfg.base.SEQUENCE_LENGTH
        T, C, H, W = event_data.shape
        max_possible = T - seq_len - start_day - 1
        sim_days = max(1, min(max_possible, self.cfg.sim_days_cap))
        if sim_days <= 0:
            raise ValueError("Fire event too short for simulation.")

        init_seq, weather, ground_truth = self.loader.prepare_simulation_data(
            event_data, start_day=start_day, max_future_days=sim_days
        )
        # Baseline
        baseline = self._simulate(init_seq, weather, sim_days)
        base_area = sum(_area_px(b, self.cfg.fire_threshold) for b in baseline)

        # Perturbations
        exp_dir, principle, (p0, p1), steps = self.cfg.hypotheses[var_name]
        perts = np.linspace(p0, p1, steps)
        preds, areas, changes = [], [], []

        for p in tqdm(perts, desc=f"Perturbing {var_name}"):
            p_seq = self._apply_factor(init_seq, feat_idx, p)
            p_wth = self._apply_factor(weather, feat_idx, p) if weather is not None else None
            pred = self._simulate(p_seq, p_wth, sim_days)
            a = sum(_area_px(x, self.cfg.fire_threshold) for x in pred)
            preds.append(pred); areas.append(float(a))
            changes.append(((a - base_area) / max(base_area, 1.0)) * 100.0)

        corr = _safe_corrcoef(perts, changes)
        slope, intercept, r_value, p_value, std_err = _safe_linregress(perts, changes)
        observed_dir = 'positive' if corr > 0 else 'negative'
        consistent = (abs(corr) > 0.1) if exp_dir == 'variable' else (exp_dir == observed_dir)

        return {
            'variable_name': var_name,
            'simulation_days': sim_days,
            'baseline_prediction': baseline,
            'ground_truth': ground_truth,
            'baseline_area': float(base_area),
            'perturbations': [float(x) for x in perts],
            'predictions': preds,
            'fire_areas': areas,
            'area_changes': [float(x) for x in changes],
            'correlation': float(corr),
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'physics_consistent': bool(consistent),
            'principle': principle,
        }


# =============================== 可视化（含动态图） ===============================
class PhysicsVisualizer:
    def __init__(self, cfg: PhysicsValidationConfig):
        self.cfg = cfg

    def comparison_gif(self, result: dict, out_path: str, interval=800, fps=1.5):
        """
        2x2 动态对比：
          TL: baseline[t]
          TR: baseline[t+1]
          BL: actual[t+1]（若缺失则提示）
          BR: extreme perturbation[t+1]，并显示与 baseline 的面积差/IoU
        """
        baseline = result['baseline_prediction']
        ground_truth = result.get('ground_truth', None)
        perts = np.asarray(result['perturbations'], dtype=float)
        preds = result['predictions']
        th = float(self.cfg.fire_threshold)

        if len(baseline) < 2:
            print("Skip GIF: not enough baseline frames.")
            return

        # 选“极端”扰动（最后一个）
        extreme_idx = len(perts) - 1 if len(perts) > 0 else None
        if extreme_idx is None or extreme_idx >= len(preds):
            print("Skip GIF: no perturbation predictions.")
            return
        perturbed = preds[extreme_idx]

        # 对齐长度
        L = min(len(baseline), len(perturbed))
        if ground_truth is not None:
            L = min(L, len(ground_truth))
        if L < 2:
            print("Skip GIF: frames < 2 after alignment.")
            return
        baseline = _pad_hold_last(baseline, L)
        perturbed = _pad_hold_last(perturbed, L)
        if ground_truth is not None:
            ground_truth = _pad_hold_last(ground_truth, L)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        var = result['variable_name']
        title = f"{var} – Baseline vs Perturbed vs Actual"

        def draw(ax, img, ttl):
            ax.clear(); ax.axis('off')
            if img is None:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            else:
                ax.imshow(img, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(ttl, fontweight='bold')

        def animate(t):
            for ax in axes: ax.clear(); ax.axis('off')
            cur = baseline[t]
            bl_next = baseline[t+1]
            pt_next = perturbed[t+1]
            gt_next = ground_truth[t+1] if ground_truth is not None else None

            draw(axes[0], cur, f"Current Fire - Day {t+1}")
            draw(axes[1], bl_next, f"Predicted (Baseline) - Day {t+2}")
            draw(axes[2], gt_next, f"Actual - Day {t+2}" if gt_next is not None else "Actual - (missing)")

            axes[3].clear(); axes[3].axis('off')
            axes[3].imshow(pt_next, cmap='Oranges', vmin=0, vmax=1, interpolation='nearest')
            axes[3].set_title(f"Predicted (Perturbed) - Day {t+2}", fontweight='bold')

            # 右侧统计
            area_bl = _area_px(bl_next, th)
            area_pt = _area_px(pt_next, th)
            delta = area_pt - area_bl
            pct = (delta / max(area_bl, 1)) * 100.0
            if gt_next is not None:
                iou_bl = _iou(bl_next, gt_next, th)
                iou_pt = _iou(pt_next, gt_next, th)
            else:
                iou_bl = iou_pt = 0.0

            box = Rectangle((0.62, 0.05), 0.33, 0.28, transform=axes[3].transAxes,
                            facecolor='lightgray', alpha=0.9, edgecolor='black')
            axes[3].add_patch(box)
            txt = (
                f"ΔArea (pert-baseline): {delta:+d} px ({pct:+.1f}%)\n"
                f"IoU vs Actual:\n"
                f"  Baseline : {iou_bl:.3f}\n"
                f"  Perturbed: {iou_pt:.3f}\n"
                f"Threshold: {th:.2f}"
            )
            axes[3].text(0.635, 0.285, txt, transform=axes[3].transAxes,
                         fontsize=11, fontfamily='monospace', va='top')

            fig.suptitle(f"{title}\nDay {t+1} → {t+2}", fontsize=16, fontweight='bold')
            return []

        frames = L - 1
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False, repeat=True)
        try:
            anim.save(out_path, writer='pillow', fps=1.5)
            print(f"  [GIF] Saved: {out_path} (frames={frames})")
        except Exception as e:
            print(f"  [GIF WARNING] {e}; save key frames as PNG.")
            for f in [0, frames//3, 2*frames//3, frames-1]:
                animate(f)
                plt.savefig(out_path.replace('.gif', f'_frame_{f}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def fire_evolution_plot(self, result: dict, out_path: str):
        var = result['variable_name']
        perts = np.asarray(result['perturbations'], dtype=float)
        preds = result['predictions']
        baseline = result['baseline_prediction']
        th = float(self.cfg.fire_threshold)

        # 面积时间序列
        base_areas = [_area_px(x, th) for x in baseline]
        days = np.arange(1, len(base_areas)+1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes.flatten()

        ax1.plot(days, base_areas, 'k-', lw=3, label='Baseline', alpha=0.85)
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(perts)))
        for i, (p, c) in enumerate(zip(perts, colors)):
            seq = preds[i] if i < len(preds) else []
            areas = [_area_px(x, th) for x in seq]
            areas = _pad_hold_last(areas, len(base_areas))
            ax1.plot(days, areas, color=c, lw=1.5, alpha=0.8, label=f'{p:+.1%}' if (i % 2 == 0 or len(perts) <= 5) else "")
        ax1.set_title('Fire Area Evolution', fontweight='bold'); ax1.set_xlabel('Day'); ax1.set_ylabel('Pixels')
        ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(days, np.cumsum(base_areas), 'k-', lw=3, label='Baseline', alpha=0.85)
        for i, (p, c) in enumerate(zip(perts, colors)):
            seq = preds[i] if i < len(preds) else []
            areas = _pad_hold_last([_area_px(x, th) for x in seq], len(base_areas))
            ax2.plot(days, np.cumsum(areas), color=c, lw=1.5, alpha=0.8, label=f'{p:+.1%}' if (i % 2 == 0 or len(perts) <= 5) else "")
        ax2.set_title('Cumulative Fire Impact', fontweight='bold'); ax2.set_xlabel('Day'); ax2.set_ylabel('Pixels')
        ax2.grid(True, alpha=0.3); ax2.legend()

        # 响应曲线
        tot_base = float(np.sum(base_areas))
        xs, ys = [], []
        for i, p in enumerate(perts):
            seq = preds[i] if i < len(preds) else []
            areas = [_area_px(x, th) for x in seq]
            ys.append(float(np.sum(areas)))
            xs.append(float(p))
        if len(xs) >= 2:
            ax3.plot(xs, ys, 'bo-', lw=2, ms=8)
            ax3.axhline(y=tot_base, color='red', ls='--', lw=2, label='Baseline Total')
        ax3.set_title('Variable Response Curve', fontweight='bold'); ax3.set_xlabel(f'{var} factor'); ax3.set_ylabel('Total pixels')
        ax3.grid(True, alpha=0.3); ax3.legend()

        # 最后一日差分（极端 vs 基线）
        last = len(baseline) - 1
        if last >= 0 and len(perts) > 0:
            extreme_idx = len(perts) - 1
            if extreme_idx < len(preds) and last < len(preds[extreme_idx]):
                ex = preds[extreme_idx][last]
                bl = baseline[last]
                diff = ex - bl
                im = ax4.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                ax4.set_title(f'Final-Day Difference (Extreme vs Baseline)', fontweight='bold')
                ax4.axis('off'); c = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04); c.set_label('Δ prob')

        plt.suptitle(f'{var} Fire Evolution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout(); plt.subplots_adjust(top=0.9)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [PNG] Saved: {out_path}")

    def response_curve_plot(self, result: dict, out_path: str):
        var = result['variable_name']
        perts = np.asarray(result['perturbations'], dtype=float)
        changes = np.asarray(result['area_changes'], dtype=float)
        corr = float(result.get('correlation', 0.0))

        baseline = result['baseline_prediction']
        preds = result['predictions']
        th = float(self.cfg.fire_threshold)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot(perts, changes, 'bo-', lw=2, ms=8)
        ax1.axhline(0, color='gray', ls='--', alpha=0.5)
        ax1.axvline(0, color='gray', ls='--', alpha=0.5)
        if len(perts) > 2 and np.std(perts) > 1e-12:
            z = np.polyfit(perts, changes, 1)
            p = np.poly1d(z)
            ax1.plot(perts, p(perts), 'r--', alpha=0.8, lw=2)
        ax1.set_title(f'{var} Response Curve', fontweight='bold'); ax1.set_xlabel(f'{var} factor'); ax1.set_ylabel('Area change (%)')
        ax1.grid(True, alpha=0.3)
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes,
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        base_by_day = [_area_px(x, th) for x in baseline]
        ax2.plot(np.arange(1, len(base_by_day)+1), base_by_day, 'k-', lw=3, label='Baseline', alpha=0.85)
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(perts)))
        for i, (pert, color) in enumerate(zip(perts, colors)):
            seq = preds[i] if i < len(preds) else []
            areas = [_area_px(x, th) for x in seq]; areas = _pad_hold_last(areas, len(base_by_day))
            ax2.plot(np.arange(1, len(base_by_day)+1), areas, color=color, lw=1.5, alpha=0.8,
                     label=f'{pert:+.1%}' if (i % 2 == 0 or len(perts) <= 5) else "")
        ax2.set_title('Per-Day Fire Area', fontweight='bold'); ax2.set_xlabel('Day'); ax2.set_ylabel('Pixels')
        ax2.grid(True, alpha=0.3); ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"  [PNG] Saved: {out_path}")


# =============================== 结果汇总与报告 ===============================
class PhysicsAnalyzer:
    def __init__(self, cfg: PhysicsValidationConfig):
        self.cfg = cfg

    def results_table(self, results, out_xlsx):
        rows = []
        for r in results:
            var = r['variable_name']
            exp_dir, _, rng, _ = self.cfg.hypotheses[var]
            maxchg = max(r['area_changes']) if r['area_changes'] else 0.0
            minchg = min(r['area_changes']) if r['area_changes'] else 0.0
            strength = "Strong" if abs(r['correlation']) > 0.7 else ("Moderate" if abs(r['correlation']) > 0.4 else "Weak")
            rows.append({
                'Physical_Variable': var,
                'Simulation_Days': r.get('simulation_days', 'N/A'),
                'Perturbation_Range': f"{rng[0]:+.1%} to {rng[1]:+.1%}",
                'Expected_Direction': exp_dir,
                'Observed_Direction': 'positive' if r['correlation'] > 0 else 'negative',
                'Correlation_Coeff': f"{r['correlation']:.4f}",
                'P_Value': f"{r['p_value']:.4f}",
                'Max_Area_Change': f"{maxchg:+.1f}%",
                'Min_Area_Change': f"{minchg:+.1f}%",
                'Response_Strength': strength,
                'Physics_Consistent': "Yes" if r['physics_consistent'] else "No",
                'Physical_Principle': r.get('principle', ''),
            })
        df = pd.DataFrame(rows)
        csv_path = out_xlsx.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        try:
            df.to_excel(out_xlsx, index=False, sheet_name='Physics Validation Results')
            print(f"[TABLE] Saved: {out_xlsx}")
        except Exception as e:
            print(f"[TABLE WARNING] {e}; CSV saved: {csv_path}")
        return df

    def summary(self, results, out_json):
        corrs = [float(r['correlation']) for r in results] if results else [0.0]
        ok = sum(1 for r in results if r['physics_consistent'])
        n = max(1, len(results))
        obj = {
            'experiment_info': {
                'variables_tested': len(results),
                'fire_threshold': self.cfg.fire_threshold,
                'timestamp': str(np.datetime64('now'))
            },
            'overall_performance': {
                'physics_consistency_rate': f"{ok}/{n} ({ok/n*100:.1f}%)",
                'mean_correlation': float(np.mean(corrs)),
                'median_correlation': float(np.median(corrs)),
                'strong_responses': int(np.sum([abs(c) > 0.7 for c in corrs])),
                'moderate_responses': int(np.sum([(0.4 < abs(c) <= 0.7) for c in corrs])),
                'weak_responses': int(np.sum([abs(c) <= 0.4 for c in corrs])),
            },
            'by_variable': {
                r['variable_name']: {
                    'simulation_days': r.get('simulation_days', 0),
                    'correlation': float(r['correlation']),
                    'p_value': float(r['p_value']),
                    'max_fire_area_change_%': max(r['area_changes']) if r['area_changes'] else 0.0,
                    'min_fire_area_change_%': min(r['area_changes']) if r['area_changes'] else 0.0,
                    'baseline_total_area': float(r['baseline_area']),
                    'physics_consistent': bool(r['physics_consistent']),
                } for r in results
            }
        }
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARY] Saved: {out_json}")
        return obj

    def overview_dashboard(self, results, out_png):
        if not results:
            return
        names = [r['variable_name'] for r in results]
        corrs = [float(r['correlation']) for r in results]
        cons = [bool(r['physics_consistent']) for r in results]
        maxchg = [max(r['area_changes']) if r['area_changes'] else 0.0 for r in results]
        minchg = [min(r['area_changes']) if r['area_changes'] else 0.0 for r in results]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        colors = ['green' if c else 'red' for c in cons]
        bars = ax1.bar(range(len(names)), corrs, color=colors, alpha=0.75)
        ax1.set_title('Correlation by Variable', fontweight='bold'); ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.axhline(0, color='gray', ls='--', alpha=0.5)
        for b,c in zip(bars, corrs):
            h = b.get_height()
            ax1.text(b.get_x()+b.get_width()/2., h + (0.02 if h>=0 else -0.05),
                     f'{c:.3f}', ha='center', va='bottom' if h>=0 else 'top', fontsize=9)

        ok = sum(cons); bad = len(cons)-ok
        if len(cons) > 0:
            ax2.pie([ok, bad], labels=['Consistent','Inconsistent'],
                    colors=['lightgreen','lightcoral'], autopct='%1.1f%%', startangle=90)
            ax2.set_title('Physics Consistency', fontweight='bold')

        x = np.arange(len(names)); w = 0.35
        ax3.bar(x - w/2, maxchg, w, label='Max %', color='orange', alpha=0.75)
        ax3.bar(x + w/2, minchg, w, label='Min %', color='blue', alpha=0.75)
        ax3.set_title('Response Magnitude', fontweight='bold'); ax3.grid(True, alpha=0.3)
        ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=45, ha='right'); ax3.legend(); ax3.axhline(0, color='gray')

        ranges = [maxchg[i]-minchg[i] for i in range(len(names))]
        sc = ax4.scatter([abs(c) for c in corrs], ranges, c=colors, s=100, alpha=0.8)
        for i,nm in enumerate(names):
            ax4.annotate(nm, (abs(corrs[i]), ranges[i]), xytext=(5,5),
                         textcoords='offset points', fontsize=9)
        ax4.set_title('Abs(Correlation) vs Response Range', fontweight='bold'); ax4.grid(True, alpha=0.3)
        plt.suptitle('Physics Validation Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()
        print(f"[DASHBOARD] Saved: {out_png}")


# =============================== Runner（主流程，一键式） ===============================
class PhysicsValidationRunner:
    def __init__(self, model_path, cfg: PhysicsValidationConfig, out_dir: str):
        self.cfg = cfg
        self.cfg.output_dir = out_dir
        self.loader = FixedFireEventLoader(cfg.base)
        self.sim = FixedFireSpreadSimulator(model_path, cfg.base)
        self.exp = ControlledPerturbationExperiment(self.sim, self.loader, cfg)
        self.viz = PhysicsVisualizer(cfg)
        self.ana = PhysicsAnalyzer(cfg)
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        print(f"[INIT] Simulator on {self.sim.device} | Params: {sum(p.numel() for p in self.sim.model.parameters()):,}")

    def discover_event(self):
        patterns = [
            'data/processed/2020/fire_24461899.hdf5',
            'data/processed/2019/*.hdf5',
            'data/processed/*/*.hdf5',
            'fire_*.hdf5',
            '*.hdf5',
        ]
        for p in patterns:
            fs = glob.glob(p)
            if fs:
                return fs[0]
        return None

    def create_demo(self, out_path="demo_fire_event.hdf5"):
        # 生成简单 demo
        T, H, W = 25, 128, 128
        C = len(self.cfg.base.FEATURE_NAMES)
        data = np.random.randn(T, C, H, W).astype(np.float32)
        fire_ch = len(self.cfg.base.BEST_FEATURES) - 1
        ch, cw = H//2, W//2
        for t in range(T):
            size = int(3 + 1.2*t)
            hs, he = max(0, ch-size), min(H, ch+size)
            ws, we = max(0, cw-size), min(W, cw+size)
            if he>hs and we>ws:
                data[t, fire_ch, hs:he, ws:we] = np.random.rand(he-hs, we-ws).astype(np.float32)*0.7
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('data', data=data)
        print(f"[DEMO] Created: {out_path}")
        return out_path

    def run(self, fire_event_path=None, start_day=0, interval=800, fps=1.5):
        if fire_event_path is None:
            fire_event_path = self.discover_event()
            if fire_event_path:
                print(f"[EVENT] Using: {fire_event_path}")
        if fire_event_path is None:
            fire_event_path = str(Path(self.cfg.output_dir) / "demo_fire_event.hdf5")
            self.create_demo(fire_event_path)

        print("="*60); print("PHYSICS VALIDATION (One-Click)"); print("="*60)
        event = self.loader.load_fire_event(fire_event_path)
        T, C, H, W = event.shape
        print(f"[EVENT] Shape: T={T}, C={C}, H={H}, W={W}")

        variables = self.cfg.testable_variables()
        print(f"[VARS] Testing {len(variables)} variables: {variables}")

        results_all = []
        for var in variables:
            print("\n" + "="*40 + f"\nTEST: {var}\n" + "="*40)
            try:
                r = self.exp.run_on_variable(event, var, start_day=start_day)
                results_all.append(r)

                vdir = Path(self.cfg.output_dir) / var
                vdir.mkdir(exist_ok=True, parents=True)

                gif_path = str(vdir / f"{var}_comparison.gif")
                evo_path = str(vdir / f"{var}_fire_evolution.png")
                curve_path = str(vdir / f"{var}_response_curve.png")
                data_path = str(vdir / f"{var}_data.json")

                self.viz.comparison_gif(r, gif_path, interval=interval, fps=fps)
                self.viz.fire_evolution_plot(r, evo_path)
                self.viz.response_curve_plot(r, curve_path)

                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)

                print(f"  Result: r={r['correlation']:+.4f}, consistent={r['physics_consistent']}, days={r['simulation_days']}")
            except Exception as e:
                print(f"[ERROR] {var}: {e}")
                import traceback; traceback.print_exc()

        if not results_all:
            print("[WARN] No results produced.")
            return

        # 汇总
        table_xlsx = str(Path(self.cfg.output_dir) / "physics_validation_results.xlsx")
        summary_json = str(Path(self.cfg.output_dir) / "summary_statistics.json")
        report_txt  = str(Path(self.cfg.output_dir) / "physics_validation_report.txt")
        dash_png    = str(Path(self.cfg.output_dir) / "overview_dashboard.png")

        df = self.ana.results_table(results_all, table_xlsx)
        sm = self.ana.summary(results_all, summary_json)

        # 简短文字报告
        ok = sum(1 for r in results_all if r['physics_consistent']); n = len(results_all)
        report = [
            "WILDFIRE PHYSICS CONSISTENCY REPORT",
            "===================================",
            f"Variables tested: {n}",
            f"Physics consistent: {ok}/{n} ({ok/n*100:.1f}%)",
            f"Mean correlation: {np.mean([r['correlation'] for r in results_all]):+.4f}",
            ""
        ]
        for r in results_all:
            exp_dir = self.cfg.hypotheses[r['variable_name']][0]
            obs_dir = 'positive' if r['correlation'] > 0 else 'negative'
            report += [
                f"{r['variable_name']}:",
                f"  Expected: {exp_dir:9s} | Observed: {obs_dir:9s} | r={r['correlation']:+.4f} | p={r['p_value']:.4f}",
                f"  Max ΔArea: {max(r['area_changes']):+.1f}% | Min ΔArea: {min(r['area_changes']):+.1f}%",
                ""
            ]
        with open(report_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print(f"[REPORT] Saved: {report_txt}")

        self.ana.overview_dashboard(results_all, dash_png)

        print("\n" + "="*60 + "\nDONE. Results in:", self.cfg.output_dir, "\n" + "="*60)


# =============================== CLI ===============================
def main():
    ap = argparse.ArgumentParser("Physics Validation Experiment - One-Click")
    ap.add_argument('--model', type=str, default='best_fire_model_official.pth', help='Path to trained model')
    ap.add_argument('--fire_event', type=str, default=None, help='Path to HDF5 fire event; auto-discover if not set')
    ap.add_argument('--start_day', type=int, default=0, help='Start day in the event')
    ap.add_argument('--output_dir', type=str, default='physics_validation', help='Output root dir (same结构)')
    ap.add_argument('--interval', type=int, default=800, help='GIF frame interval (ms)')
    ap.add_argument('--fps', type=float, default=1.5, help='GIF frames per second')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    cfg = PhysicsValidationConfig()
    runner = PhysicsValidationRunner(args.model, cfg, args.output_dir)
    runner.run(fire_event_path=args.fire_event, start_day=args.start_day,
               interval=args.interval, fps=args.fps)


if __name__ == "__main__":
    main()
