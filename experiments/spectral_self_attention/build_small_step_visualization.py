"""Build the inline HTML summary for the small-step audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def best_continuation(data_dir: Path, tag: str, label: str) -> dict[str, object]:
    deep = data_dir / f"small_step_extension_deep_{tag}_{label}.json"
    extension = data_dir / f"small_step_extension_{tag}_{label}.json"
    source = (
        deep
        if deep.exists()
        else extension
        if extension.exists()
        else data_dir / f"small_step_continuation_{tag}_{label}.json"
    )
    return json.loads(source.read_text())


def survival_payload(data_dir: Path) -> list[dict[str, object]]:
    output = []
    for family in (1, 2, 3, 4):
        for label in ("p3", "p4", "chaos"):
            source = best_continuation(data_dir, f"f{family}", label)
            for index, ratio in enumerate(source["settings"]["ratios"]):
                points = [record["trace"][index] for record in source["records"]]
                output.append(
                    {
                        "family": family,
                        "label": label,
                        "ratio": round(float(ratio), 9),
                        "moving": round(
                            sum(
                                float(point["motion_per_normalized_time"]) >= 1e-3
                                for point in points
                            )
                            / len(points),
                            5,
                        ),
                    }
                )
    return output


def transition_payload(data_dir: Path) -> list[dict[str, float]]:
    fine = data_dir / "selected_bifurcation_f4_p3_chaos.json"
    if fine.exists():
        source = json.loads(fine.read_text())
        return [
            {
                "ratio": round(float(point["ratio"]), 9),
                "motion": round(float(point["motion_per_normalized_time"]), 5),
                "geometry": round(float(point["gram_variation"]), 5),
                "lyapunov": round(float(point["lyapunov_per_normalized_time"]), 5),
            }
            for point in source["downward"]
        ]
    source = best_continuation(data_dir, "f4", "p3")
    selected = [
        record
        for record in source["records"]
        if int(record["identity"]["n_tokens"]) == 3
        and int(record["identity"]["source_model_index"]) == 2612
    ][0]
    return [
        {
            "ratio": round(float(point["ratio"]), 9),
            "motion": round(float(point["motion_per_normalized_time"]), 5),
            "geometry": round(float(point["gram_variation"]), 5),
            "lyapunov": round(float(point["lyapunov_per_normalized_time"]), 5),
        }
        for point in selected["trace"]
    ]


def phase_payload(data_dir: Path, name: str) -> list[list[float]]:
    source = json.loads((data_dir / f"continuous_trace_{name}.json").read_text())
    points = np.asarray(source["relative_angles"], dtype=float)[::10, :2]
    return np.round(points, 4).tolist()


def headline_payload(data_dir: Path) -> dict[str, object]:
    final = json.loads((data_dir / "small_step_final_results.json").read_text())
    small_totals = {
        int(row["family"]): int(row["records"])
        for row in final["stability_audited_direct_ode_by_family"]
    }
    high_totals = {
        int(row["family"]): int(row["records"])
        for row in final["high_token_stability_audited_direct_ode_by_family"]
    }
    small = {
        str(row["family"]): round(
            100.0 * float(row["still_moving"]) / small_totals[int(row["family"])],
            2,
        )
        for row in final["long_time_strict_summaries"]["main"][
            "corrected_by_family"
        ]
    }
    high = {
        str(row["family"]): round(
            100.0 * float(row["still_moving"]) / high_totals[int(row["family"])],
            2,
        )
        for row in final["long_time_strict_summaries"]["high_token"][
            "corrected_by_family"
        ]
    }
    basins = []
    basin_specs = (
        ("chaos fort · 3", "continuous_basin_f4_strong_chaos.json"),
        ("chaos faible · 3", "continuous_basin_f2_weak_chaos.json"),
        ("attention uniforme · 4", "continuous_basin_beta0_f3_hyperchaos.json"),
        ("chimère · 8", "continuous_basin_highn_f4_n8_i1632.json"),
    )
    for label, filename in basin_specs:
        summary = json.loads((data_dir / filename).read_text())["summary"]
        basins.append(
            {
                "label": label,
                "strict": int(summary.get("internal_positive_lyapunov", 0)),
                "recurrent": int(summary.get("internal_recurrent", 0)),
                "internal": int(summary.get("internal_unresolved", 0)),
                "fixed": int(summary.get("fixed", 0)),
                "slow": int(summary.get("slow_or_unresolved", 0)),
            }
        )
    spectra = final["full_lyapunov_spectra"]
    spectrum_specs = (
        ("chaos fort · 3", "type4_strong_chaos"),
        ("chaos faible · 3", "type2_weak_chaos"),
        ("uniforme · 4", "type3_beta0_intermittent_chaos_antilock"),
        ("chimère · 8", "type4_eight_token_hyperchaos_antilock"),
    )
    return {
        "census": {"small": small, "high": high},
        "basins": basins,
        "spectra": [
            {"label": label, "values": [round(float(value), 5) for value in spectra[key]]}
            for label, key in spectrum_specs
        ],
    }


def convergence_payload(data_dir: Path) -> dict[str, object]:
    main = json.loads(
        (data_dir / "finite_horizon_convergence_main_T10.json").read_text()
    )
    high = json.loads(
        (data_dir / "finite_horizon_convergence_highn_T10.json").read_text()
    )
    richardson = json.loads(
        (data_dir / "richardson_finite_horizon_main_T2.json").read_text()
    )
    raw_cells: dict[float, list[float]] = {}
    for path in sorted(
        data_dir.glob("random_model_finite_horizon_deep_T10_raw_f*_768.json")
    ):
        payload = json.loads(path.read_text())
        for row in payload["records"]:
            ratio = float(row["step_ratio"])
            raw_cells.setdefault(ratio, []).append(
                float(row["angle_error"]["median"])
            )
    off_grid = []
    replication_paths = sorted(
        data_dir.glob(
            "random_model_finite_horizon_offgrid*_T10_raw_all_3072.json"
        )
    )
    time20 = data_dir / "random_model_finite_horizon_deep_T20_raw_all_1536.json"
    if time20.exists():
        replication_paths.append(time20)
    for path in replication_paths:
        payload = json.loads(path.read_text())
        cells: dict[float, list[float]] = {}
        for row in payload["records"]:
            ratio = float(row["step_ratio"])
            cells.setdefault(ratio, []).append(
                float(row["angle_error"]["median"])
            )
        off_grid.append(
            [
                {"ratio": ratio, "median": float(np.median(values))}
                for ratio, values in sorted(cells.items(), reverse=True)
            ]
        )
    return {
        "small": [
            {
                "ratio": float(row["step_ratio"]),
                "median": float(row["median_error"]),
                "q90": float(row["q90_error"]),
            }
            for row in main["aggregate"]
        ],
        "high": [
            {
                "ratio": float(row["step_ratio"]),
                "median": float(row["median_error"]),
                "q90": float(row["q90_error"]),
            }
            for row in high["aggregate"]
        ],
        "richardson": [
            {
                "ratio": float(row["fine_step_ratio"]),
                "median": float(row["richardson_error"]["median"]),
            }
            for row in richardson["aggregate"]
        ],
        "unscreened": [
            {
                "ratio": ratio,
                "median": float(np.median(values)),
            }
            for ratio, values in sorted(raw_cells.items(), reverse=True)
        ],
        "offGrid": off_grid,
    }


def basin_scaling_payload(data_dir: Path) -> dict[str, object]:
    series = []
    orders = []
    for label, model_index in (("A", 2639), ("B", 1209)):
        coarse = json.loads(
            (
                data_dir
                / f"basin_partition_mismatch_scaling_f1_n4_i{model_index}.json"
            ).read_text()
        )
        deep = json.loads(
            (
                data_dir
                / f"basin_partition_mismatch_scaling_f1_n4_i{model_index}_deep.json"
            ).read_text()
        )
        rows = coarse["records"][:4] + deep["records"]
        ratios = np.asarray([float(row["step_ratio"]) for row in rows])
        mismatches = np.asarray(
            [float(row["partition_mismatch_fraction"]) for row in rows]
        )
        orders.append(float(np.polyfit(np.log(ratios), np.log(mismatches), 1)[0]))
        series.append(
            {
                "label": label,
                "records": [
                    {
                        "ratio": float(row["step_ratio"]),
                        "mismatch": float(row["partition_mismatch_fraction"]),
                        "distribution": float(
                            row["partition_distribution_total_variation"]
                        ),
                    }
                    for row in rows
                ],
            }
        )
    return {
        "orders": [round(value, 3) for value in orders],
        "series": series,
    }


def build_fragment(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"))
    return f"""<div id="small-step-audit" class="vstack gap-4">
  <section aria-label="Survie quand la couche devient petite">
    <div class="viz-row text-small" aria-hidden="true">
      <span><span class="audit-swatch audit-s1"></span>cycle 3</span>
      <span><span class="audit-swatch audit-s2"></span>cycle 4</span>
      <span><span class="audit-swatch audit-s3"></span>chaos initial</span>
    </div>
    <canvas id="audit-survival" role="img" aria-label="Pourcentage d'attracteurs encore en mouvement selon le pas, dans les quatre types"></canvas>
  </section>
  <section aria-label="Transformation du cycle 3 en chaos">
    <canvas id="audit-transition" role="img" aria-label="Mouvement, changement de géométrie et séparation des trajectoires pendant la réduction du pas"></canvas>
  </section>
  <section aria-label="Convergence au flot continu à temps dix">
    <div class="viz-row text-small" aria-hidden="true">
      <span><span class="audit-swatch audit-s1"></span>médiane · 1–4 tokens</span>
      <span><span class="audit-swatch audit-s2"></span>médiane · 8–16 tokens</span>
      <span><span class="audit-swatch audit-s3"></span>erreur dominante annulée</span>
      <span><span class="audit-swatch audit-s4"></span>24 cellules brutes · 4 grilles</span>
    </div>
    <canvas id="audit-convergence" role="img" aria-label="Erreur à temps dix face au flot continu, décroissant proportionnellement au pas pour les petites et grandes populations"></canvas>
  </section>
  <section aria-label="Déplacement des bassins de groupes de tokens">
    <div class="viz-row text-small" aria-hidden="true">
      <span><span class="audit-swatch audit-s1"></span>départs avec une autre partition</span>
      <span><span class="audit-swatch audit-s2"></span>réplication indépendante</span>
      <span><span class="audit-swatch audit-s3"></span>distribution · paysage A</span>
      <span><span class="audit-swatch audit-s4"></span>distribution · paysage B</span>
    </div>
    <canvas id="audit-basinscaling" role="img" aria-label="Fraction de départs dont les tokens finissent dans des groupes différents, décroissant presque proportionnellement au pas sur 65 536 départs"></canvas>
  </section>
  <section aria-label="Survie par structure et nombre de tokens">
    <canvas id="audit-census" role="img" aria-label="Pourcentage de candidats encore mobiles dans l'équation continue pour les quatre structures"></canvas>
  </section>
  <section aria-label="Bassins globaux et spectres">
    <div class="audit-summary-grid">
      <div>
        <div class="viz-row text-small" aria-hidden="true">
          <span><span class="audit-swatch audit-s1"></span>chaos strict</span>
          <span><span class="audit-swatch audit-s2"></span>récurrent</span>
          <span><span class="audit-swatch audit-s3"></span>interne lent</span>
          <span><span class="audit-swatch audit-s4"></span>fixe</span>
        </div>
        <canvas id="audit-basins" role="img" aria-label="Répartition des destins depuis des départs aléatoires"></canvas>
      </div>
      <div><canvas id="audit-spectra" role="img" aria-label="Spectres complets des quatre attracteurs représentatifs"></canvas></div>
    </div>
  </section>
  <section aria-label="Espaces des angles relatifs">
    <div class="audit-phase-grid">
      <div><div class="text-small">cycle stable · type 4</div><canvas id="audit-cycle" role="img" aria-label="Courbe fermée du cycle stable dans le plan des angles relatifs"></canvas></div>
      <div><div class="text-small">chaos faible · type 2</div><canvas id="audit-chaos2" role="img" aria-label="Attracteur chaotique faible dans le plan des angles relatifs"></canvas></div>
      <div><div class="text-small">chaos fort · type 4</div><canvas id="audit-chaos4" role="img" aria-label="Attracteur chaotique fort dans le plan des angles relatifs"></canvas></div>
      <div><div class="text-small">attention uniforme · type 3</div><canvas id="audit-beta0" role="img" aria-label="Attracteur intermittent avec poids d'attention uniformes"></canvas></div>
    </div>
  </section>
</div>
<style>
#small-step-audit {{ color: var(--foreground); width: 100%; }}
#small-step-audit canvas {{ display: block; width: 100%; height: auto; }}
#small-step-audit .audit-phase-grid {{ display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 16px; }}
#small-step-audit .audit-summary-grid {{ display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 20px; }}
#small-step-audit .audit-swatch {{ display:inline-block; width:12px; height:3px; margin-inline-end:5px; vertical-align:middle; background:var(--viz-series-1); }}
#small-step-audit .audit-s2 {{ background:var(--viz-series-2); }}
#small-step-audit .audit-s3 {{ background:var(--viz-series-3); }}
#small-step-audit .audit-s4 {{ background:var(--viz-series-4); }}
@media (max-width: 700px) {{ #small-step-audit .audit-phase-grid {{ grid-template-columns: repeat(2,minmax(0,1fr)); }} }}
@media (max-width: 560px) {{ #small-step-audit .audit-phase-grid, #small-step-audit .audit-summary-grid {{ grid-template-columns: 1fr; }} }}
</style>
<script>
(() => {{
  const root=document.getElementById('small-step-audit');
  const data={encoded};
  const css=getComputedStyle(root);
  const colors=[css.getPropertyValue('--viz-series-1').trim(),css.getPropertyValue('--viz-series-2').trim(),css.getPropertyValue('--viz-series-3').trim(),css.getPropertyValue('--viz-series-4').trim()];
  const fg=css.getPropertyValue('--foreground').trim();
  const muted=css.getPropertyValue('--muted-foreground').trim();
  const border=css.getPropertyValue('--border').trim();
  const font=css.getPropertyValue('--font-size-base').trim()+' system-ui,sans-serif';
  function setup(canvas,ratio){{
    const width=Math.max(300,Math.floor(canvas.getBoundingClientRect().width));
    canvas.width=width*devicePixelRatio;canvas.height=Math.floor(width*ratio)*devicePixelRatio;
    const c=canvas.getContext('2d');c.scale(devicePixelRatio,devicePixelRatio);c.font=font;c.lineJoin='round';c.lineCap='round';
    return [c,width,Math.floor(width*ratio)];
  }}
  function line(c,points,color,width=2){{c.beginPath();points.forEach((p,i)=>i?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1]));c.strokeStyle=color;c.lineWidth=width;c.stroke();}}
  function survival(){{
    const canvas=document.getElementById('audit-survival');const [c,w,h]=setup(canvas,.62);const gap=34,top=28,left=42,right=14,bottom=34;
    const pw=(w-left-right-gap)/2,ph=(h-top-bottom-gap)/2;const maxDepth=Math.max(...data.survival.map(d=>Math.log2(1/d.ratio))),maxY=.3;
    for(let family=1;family<=4;family++){{
      const col=(family-1)%2,row=Math.floor((family-1)/2),x0=left+col*(pw+gap),y0=top+row*(ph+gap);
      c.fillStyle=fg;c.textAlign='left';c.fillText('type '+family,x0,y0-9);
      c.strokeStyle=border;c.lineWidth=1;c.strokeRect(x0,y0,pw,ph);
      [0,.1,.2,.3].forEach(v=>{{const y=y0+ph-v/maxY*ph;c.beginPath();c.moveTo(x0,y);c.lineTo(x0+pw,y);c.strokeStyle=border;c.stroke();if(col===0){{c.fillStyle=muted;c.textAlign='right';c.fillText(Math.round(v*100)+'%',x0-5,y+4);}}}});
      ['p3','p4','chaos'].forEach((label,li)=>{{const rows=data.survival.filter(d=>d.family===family&&d.label===label);const pts=rows.map(d=>[x0+Math.log2(1/d.ratio)/maxDepth*pw,y0+ph-d.moving/maxY*ph]);line(c,pts,colors[li]);}});
      if(row===1){{[0,4,8,12].filter(v=>v<=maxDepth).forEach(v=>{{const x=x0+v/maxDepth*pw;c.fillStyle=muted;c.textAlign='center';c.fillText(v===0?'h':'h/'+Math.pow(2,v),x,y0+ph+18);}});}}
    }}
  }}
  function convergence(){{
    const canvas=document.getElementById('audit-convergence');const [c,w,h]=setup(canvas,.46);const left=58,right=22,top=20,bottom=40,pw=w-left-right,ph=h-top-bottom;
    const minY=1e-7,maxY=.2,minD=5,maxD=12.5;
    const sx=r=>left+(Math.log2(1/r)-minD)/(maxD-minD)*pw,sy=v=>top+(Math.log10(maxY)-Math.log10(v))/(Math.log10(maxY)-Math.log10(minY))*ph;
    c.strokeStyle=border;c.strokeRect(left,top,pw,ph);
    [1e-7,1e-5,1e-3,1e-1].forEach(v=>{{const y=sy(v);c.strokeStyle=border;c.beginPath();c.moveTo(left,y);c.lineTo(left+pw,y);c.stroke();c.fillStyle=muted;c.textAlign='right';c.fillText(v.toExponential(0),left-6,y+4);}});
    [5,6,8,10,12].forEach(v=>{{const x=left+(v-minD)/(maxD-minD)*pw;c.fillStyle=muted;c.textAlign='center';c.fillText('h/'+Math.pow(2,v),x,h-13);}});
    [['small',colors[0]],['high',colors[1]],['richardson',colors[2]],['unscreened',colors[3]]].forEach(([key,color])=>{{const rows=data.convergence[key];line(c,rows.map(d=>[sx(d.ratio),sy(d.median)]),color,2.5);rows.forEach(d=>{{c.fillStyle=color;c.beginPath();c.arc(sx(d.ratio),sy(d.median),3,0,2*Math.PI);c.fill();}});}});
    data.convergence.offGrid.forEach((rows,index)=>{{c.setLineDash(index===0?[6,4]:index===1?[2,4]:[8,3,2,3]);line(c,rows.map(d=>[sx(d.ratio),sy(d.median)]),colors[3],2);c.setLineDash([]);rows.forEach(d=>{{c.fillStyle=css.getPropertyValue('--background').trim();c.strokeStyle=colors[3];c.lineWidth=2;c.beginPath();c.arc(sx(d.ratio),sy(d.median),3+index*.6,0,2*Math.PI);c.fill();c.stroke();}});}});
    c.fillStyle=muted;c.textAlign='center';c.fillText('pas de couche',left+pw/2,h-2);
  }}
  function basinScaling(){{
    const canvas=document.getElementById('audit-basinscaling');const [c,w,h]=setup(canvas,.44);const left=62,right=22,top=26,bottom=42,pw=w-left-right,ph=h-top-bottom,minY=1e-5,maxY=.03,minD=5,maxD=12;
    const sx=r=>left+(Math.log2(1/r)-minD)/(maxD-minD)*pw,sy=v=>top+(Math.log10(maxY)-Math.log10(v))/(Math.log10(maxY)-Math.log10(minY))*ph;
    c.strokeStyle=border;c.strokeRect(left,top,pw,ph);
    [1e-5,1e-4,1e-3,1e-2].forEach(v=>{{const y=sy(v);c.strokeStyle=border;c.beginPath();c.moveTo(left,y);c.lineTo(left+pw,y);c.stroke();c.fillStyle=muted;c.textAlign='right';c.fillText((100*v).toFixed(v<.001?3:1)+'%',left-6,y+4);}});
    [5,7,9,11,12].forEach(v=>{{const x=left+(v-minD)/(maxD-minD)*pw;c.fillStyle=muted;c.textAlign='center';c.fillText('h/'+Math.pow(2,v),x,h-14);}});
    const specs=[['mismatch',0],['mismatch',1],['distribution',0],['distribution',1]];
    specs.forEach(([key,index],si)=>{{const pts=data.basinScaling.series[index].records.map(d=>[sx(d.ratio),sy(d[key])]);line(c,pts,colors[si],2.5);pts.forEach(p=>{{c.fillStyle=colors[si];c.beginPath();c.arc(p[0],p[1],3,0,2*Math.PI);c.fill();}});}});
    c.fillStyle=fg;c.textAlign='left';c.fillText('pentes '+data.basinScaling.orders.map(v=>v.toFixed(2)).join(' · '), left+8,top+16);c.fillStyle=muted;c.textAlign='center';c.fillText('pas de couche',left+pw/2,h-2);
  }}
  function transition(){{
    const canvas=document.getElementById('audit-transition');const [c,w,h]=setup(canvas,.48);const left=48,right=18,top=22,bottom=34,gap=14;const band=(h-top-bottom-2*gap)/3,maxDepth=Math.ceil(Math.max(...data.transition.map(d=>Math.log2(1/d.ratio))));
    const series=[['motion',2.2,'mouvement',colors[0]],['geometry',1,'géométrie interne',colors[1]],['lyapunov',.25,'séparation',colors[2]]];
    series.forEach((s,si)=>{{const y0=top+si*(band+gap);c.fillStyle=fg;c.textAlign='left';c.fillText(s[2],left,y0-6);c.strokeStyle=border;c.strokeRect(left,y0,w-left-right,band);const pts=data.transition.map(d=>[left+Math.log2(1/d.ratio)/maxDepth*(w-left-right),y0+band-Math.max(0,d[s[0]])/s[1]*band]);line(c,pts,s[3]);}});
    Array.from({{length:Math.floor(maxDepth/2)+1}},(_,i)=>2*i).forEach(v=>{{const x=left+v/maxDepth*(w-left-right);c.fillStyle=muted;c.textAlign='center';c.fillText(v===0?'h':'h/'+Math.pow(2,v),x,h-9);}});
  }}
  function census(){{
    const canvas=document.getElementById('audit-census');const [c,w,h]=setup(canvas,.4);const left=48,right=18,top=20,bottom=48,pw=w-left-right,ph=h-top-bottom,maxY=35;
    c.strokeStyle=border;c.strokeRect(left,top,pw,ph);
    [0,10,20,30].forEach(v=>{{const y=top+ph-v/maxY*ph;c.strokeStyle=border;c.beginPath();c.moveTo(left,y);c.lineTo(left+pw,y);c.stroke();c.fillStyle=muted;c.textAlign='right';c.fillText(v+'%',left-6,y+4);}});
    const group=pw/4,bar=Math.min(28,group*.25);
    for(let family=1;family<=4;family++){{const center=left+(family-.5)*group;[['small',-bar*.62,colors[0]],['high',bar*.62,colors[1]]].forEach(([key,offset,color])=>{{const value=data.headline.census[key][String(family)],height=value/maxY*ph;c.fillStyle=color;c.fillRect(center+offset-bar/2,top+ph-height,bar,height);c.fillStyle=fg;c.textAlign='center';c.fillText(value+'%',center+offset,top+ph-height-5);}});c.fillStyle=muted;c.textAlign='center';c.fillText('type '+family,center,h-23);}}
    c.fillStyle=colors[0];c.fillRect(w*.32,h-13,12,3);c.fillStyle=muted;c.textAlign='left';c.fillText('1–4 tokens',w*.32+17,h-7);c.fillStyle=colors[1];c.fillRect(w*.62,h-13,12,3);c.fillStyle=muted;c.fillText('8–16 tokens',w*.62+17,h-7);
  }}
  function basins(){{
    const canvas=document.getElementById('audit-basins');const [c,w,h]=setup(canvas,.68);const left=118,right=16,top=24,bottom=24,rowH=(h-top-bottom)/data.headline.basins.length;const keys=['strict','recurrent','internal','fixed'];
    data.headline.basins.forEach((row,ri)=>{{const total=keys.reduce((s,k)=>s+row[k],row.slow||0),y=top+ri*rowH+rowH*.25,bh=rowH*.42;c.fillStyle=fg;c.textAlign='right';c.fillText(row.label,left-7,y+bh*.7);let x=left;keys.forEach((key,ki)=>{{const value=key==='internal'?row.internal+row.slow:row[key],width=value/total*(w-left-right);c.fillStyle=colors[ki];c.globalAlpha=.82;c.fillRect(x,y,width,bh);if(width>34){{c.fillStyle=fg;c.globalAlpha=1;c.textAlign='center';c.fillText(Math.round(value/total*100)+'%',x+width/2,y+bh*.7);}}x+=width;}});c.globalAlpha=1;}});
  }}
  function spectra(){{
    const canvas=document.getElementById('audit-spectra');const [c,w,h]=setup(canvas,.68);const left=116,right=18,top=22,bottom=34,minX=-.75,maxX=.17,rowH=(h-top-bottom)/data.headline.spectra.length;
    const sx=v=>left+(v-minX)/(maxX-minX)*(w-left-right);c.strokeStyle=border;c.beginPath();c.moveTo(sx(0),top);c.lineTo(sx(0),h-bottom);c.stroke();
    data.headline.spectra.forEach((row,ri)=>{{const y=top+(ri+.5)*rowH;c.fillStyle=fg;c.textAlign='right';c.fillText(row.label,left-7,y+4);c.strokeStyle=border;c.beginPath();c.moveTo(sx(Math.min(...row.values)),y);c.lineTo(sx(Math.max(...row.values)),y);c.stroke();row.values.forEach(v=>{{c.fillStyle=v>.005?colors[2]:(v<-.005?colors[0]:muted);c.beginPath();c.arc(sx(v),y,4,0,2*Math.PI);c.fill();}});}});
    [-.7,-.4,0,.15].forEach(v=>{{c.fillStyle=muted;c.textAlign='center';c.fillText(v.toFixed(v===0?0:2),sx(v),h-10);}});
  }}
  function phase(id,points,color){{
    const canvas=document.getElementById(id);const [c,w,h]=setup(canvas,1);const pad=24;
    c.strokeStyle=border;c.strokeRect(pad,pad,w-2*pad,h-2*pad);c.fillStyle=color;c.globalAlpha=.34;
    points.forEach(p=>{{const x=pad+(p[0]+Math.PI)/(2*Math.PI)*(w-2*pad),y=h-pad-(p[1]+Math.PI)/(2*Math.PI)*(h-2*pad);c.beginPath();c.arc(x,y,1.4,0,2*Math.PI);c.fill();}});c.globalAlpha=1;
    c.fillStyle=muted;c.textAlign='center';c.fillText('angle token 2 − token 1',w/2,h-5);c.save();c.translate(9,h/2);c.rotate(-Math.PI/2);c.fillText('token 3 − token 1',0,0);c.restore();
  }}
  let lastWidth=-1;
  function draw(){{const width=Math.floor(root.getBoundingClientRect().width);if(width>0&&width===lastWidth)return;lastWidth=width;survival();transition();convergence();basinScaling();census();basins();spectra();phase('audit-cycle',data.phases.cycle,colors[0]);phase('audit-chaos2',data.phases.chaos2,colors[1]);phase('audit-chaos4',data.phases.chaos4,colors[2]);phase('audit-beta0',data.phases.beta0,colors[3]);}}
  new ResizeObserver(draw).observe(root);draw();
}})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/spectral_self_attention"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "survival": survival_payload(args.data_dir),
        "transition": transition_payload(args.data_dir),
        "headline": headline_payload(args.data_dir),
        "convergence": convergence_payload(args.data_dir),
        "basinScaling": basin_scaling_payload(args.data_dir),
        "phases": {
            "cycle": phase_payload(args.data_dir, "f4_p3_cycle"),
            "chaos2": phase_payload(args.data_dir, "f2_chaos"),
            "chaos4": phase_payload(args.data_dir, "f4_strong_chaos_relaxed2"),
            "beta0": phase_payload(args.data_dir, "beta0_f3_hyperchaos_relaxed2"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_fragment(payload))
    print(args.output)


if __name__ == "__main__":
    main()
