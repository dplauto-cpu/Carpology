""")


# ============================================
# BLOQUE 12: MAPA DE LOCALIZACIÓN
# ============================================

st.markdown("---")
st.subheader("🗺️ Localización de la muestra")

if st.button("📍 Mostrar en el mapa", type="primary"):
mapa_muestra = folium.Map(
    location=[latitud, longitud],
    zoom_start=6,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Tiles &copy; Esri &mdash; National Geographic'
)

texto_cereal = "Cebada" if prediccion == 1 else "Trigo"
color_marcador = '#8B4513' if prediccion == 1 else '#CD853F'

folium.Marker(
    location=[latitud, longitud],
    popup=f"<b>{texto_cereal}</b><br>δ13C: {d13C}‰<br>δ15N: {d15N}‰<br>{periodo}<br>{cuenca}",
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(mapa_muestra)

folium_static(mapa_muestra, width=700, height=400)
st.caption(f"📍 Marcador rojo: muestra clasificada como **{texto_cereal}** en {latitud}°N, {longitud}°E")


# ============================================
# BLOQUE 13: PIE DE PÁGINA
# ============================================

st.markdown("---")
st.caption("""