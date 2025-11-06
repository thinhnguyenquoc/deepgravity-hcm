import json

def aggregate_data():
    try:
        filename_probability = "./probability.json"
        data_probability = []
        with open(filename_probability, 'r') as file:
            data_probability = json.load(file)
        filename_grid = "./grid.json"
        data_grid = []
        with open(filename_grid, 'r') as file:
            data_grid = json.load(file)
        filename_pois = "./pois_district.json"
        data_pois = []
        with open(filename_pois, 'r') as file:
            data_pois = json.load(file)
        filename_matrix_distance = "./matrix_distance.json"
        data_matrix_distance = []
        with open(filename_matrix_distance, 'r') as file:
            data_matrix_distance = json.load(file)
        aggregated_data = []
        count = 0
        for item in data_matrix_distance:
            for distance_entry in item["distances"]:
                count += 1
               
                for p in data_grid:
                    if p["place_name"] == item["from"]:
                        population_from_cell = p["cells"]
                    if p["place_name"] == item["to"]:
                        population_to_cell = p["cells"]
                
                for p1 in population_from_cell:
                    if p1["cell_id"] == distance_entry["cell_from"]:
                        population_from = p1["pop"]
                for p2 in population_to_cell:
                    if p2["cell_id"] == distance_entry["cell_to"]:
                        population_to = p2["pop"]

           
                for d in data_pois:
                    if d["place_name"] == item["from"]:
                        pois_from = d["cell_pois"]
                    if d["place_name"] == item["to"]:
                        pois_to = d["cell_pois"]

                for d1 in pois_from:
                    if d1["cell_id"] == distance_entry["cell_from"]:
                        pois_from_cell = {
                            "cell_id": d1["cell_id"],
                            "pois_amenity": d1["pois_amenity"],
                            "pois_shop": d1["pois_shop"],
                            "pois_leisure": d1["pois_leisure"],
                            "pois_tourism": d1["pois_tourism"],
                            "pois_office": d1["pois_office"],
                            "pois_public_transport": d1["pois_public_transport"],
                        }
                        break

                for d2 in pois_to:
                    if d2["cell_id"] == distance_entry["cell_to"]:
                        pois_to_cell = {
                            "cell_id": d2["cell_id"],
                            "pois_amenity": d2["pois_amenity"],
                            "pois_shop": d2["pois_shop"],
                            "pois_leisure": d2["pois_leisure"],
                            "pois_tourism": d2["pois_tourism"],
                            "pois_office": d2["pois_office"],
                            "pois_public_transport": d2["pois_public_transport"],
                        }
                        break

                for p in data_probability:
                    if p["place_name"] == item["from"]:
                        probability_from = p
                        if distance_entry["distance_km"] == 0:
                            prob_from = probability_from["0"]
                        elif 0 < distance_entry["distance_km"] < 10:
                            prob_from = probability_from["(0, 10)"]
                        elif 10 <= distance_entry["distance_km"] < 100:
                            prob_from = probability_from["[10, 100)"]
                        else:
                            prob_from = probability_from["100+"]
                        break

                aggregated_data.append({
                    "id": count,
                    "from_district": item["from"],
                    "to_district": item["to"],
                    "cell_from": distance_entry["cell_from"],
                    "cell_to": distance_entry["cell_to"],
                    "distance_km": distance_entry["distance_km"],
                    "population_from": population_from,
                    "population_to": population_to,
                    "pois_amenity_from": pois_from_cell["pois_amenity"],
                    "pois_amenity_to": pois_to_cell["pois_amenity"],
                    "pois_shop_from": pois_from_cell["pois_shop"],
                    "pois_shop_to": pois_to_cell["pois_shop"],
                    "pois_leisure_from": pois_from_cell["pois_leisure"],
                    "pois_leisure_to": pois_to_cell["pois_leisure"],
                    "pois_tourism_from": pois_from_cell["pois_tourism"],
                    "pois_tourism_to": pois_to_cell["pois_tourism"],
                    "pois_office_from": pois_from_cell["pois_office"],
                    "pois_office_to": pois_to_cell["pois_office"],
                    "pois_public_transport_from": pois_from_cell["pois_public_transport"],
                    "pois_public_transport_to": pois_to_cell["pois_public_transport"],
                    "probability_move": prob_from
                })
        with open("./aggregated_data.json", 'w') as f:
            json.dump(aggregated_data, f, indent=4) # 'indent=4' for pretty-printing
    except Exception as e:
        print(f"Error occurred: {e}")   

aggregate_data()