import json
import schemdraw
import schemdraw.elements as elm

def create_component(comp, pos, group):
    """Create an IC element for a component at position pos.
       If 'pin_dist' exists, use it to distribute pins on four sides:
         left, right, top, bottom.
       Otherwise, assign pins to the top side for bottom group components,
       and to the right side for others.
    """
    ic_elem = elm.Ic().label(comp["name"], fontsize=12)
    
    if "pin_dist" in comp:
        dist = comp["pin_dist"]
        total_expected = sum(dist)
        pins = comp["pins"]
        if len(pins) != total_expected:
            print(f"Warning: For {comp['name']}, expected {total_expected} pins but got {len(pins)}")
        idx = 0
        # Left side pins:
        for i in range(dist[0]):
            ic_elem = ic_elem.pin(name=pins[idx], side='L', pin=str(i+1))
            idx += 1
        # Right side pins:
        for i in range(dist[1]):
            ic_elem = ic_elem.pin(name=pins[idx], side='R', pin=str(i+1))
            idx += 1
        # Top side pins:
        for i in range(dist[2]):
            ic_elem = ic_elem.pin(name=pins[idx], side='T', pin=str(i+1))
            idx += 1
        # Bottom side pins:
        for i in range(dist[3]):
            ic_elem = ic_elem.pin(name=pins[idx], side='B', pin=str(i+1))
            idx += 1
    else:
        # For peripherals without pin_dist,
        # assign all pins on the top side if the component is in the bottom group.
        if group == "bottom":
            for i, p in enumerate(comp["pins"]):
                ic_elem = ic_elem.pin(name=p, side='T', pin=str(i+1))
        else:
            # Default: assign pins on the right side.
            for i, p in enumerate(comp["pins"]):
                ic_elem = ic_elem.pin(name=p, side='R', pin=str(i+1))
    ic_elem.at(pos)
    return ic_elem

def layout_components(components):
    """
    Place the components on the drawing.
    The main MCU (Arduino_Uno) goes at (-5, 0) – shifted slightly left.
    All other peripherals are divided into two groups:
      - "left": placed to the left of the MCU.
      - "bottom": placed below the MCU.
    Returns a dictionary mapping component name to (position, group).
    """
    comp_positions = {}
    mcu_name = "Arduino_Uno"
    peripherals = []
    
    for comp in components:
        if comp["name"] == mcu_name:
            # Assign MCU to the center with group "center"
            comp_positions[mcu_name] = ((-5, 0), "center")
        else:
            peripherals.append(comp)
    
    n = len(peripherals)
    if n == 0:
        return comp_positions
    
    split_index = n // 2  # First half to left, second half to bottom.
    left_group = peripherals[:split_index]
    bottom_group = peripherals[split_index:]
    
    # Place left group: x = -20 and vertically spaced.
    for i, comp in enumerate(left_group):
        comp_positions[comp["name"]] = ((-20, 20 - i * 5), "left")
    
    # Place bottom group: y = -10 (closer to MCU) and horizontally spaced.
    for i, comp in enumerate(bottom_group):
        comp_positions[comp["name"]] = ((-5 + i * 5, -10), "bottom")
        
    return comp_positions

def draw_connections(d, comp_objs, connections):
    """
    Draw wires between pins as specified in the connections.
    Each connection has a "from": [component, pin] and "to": [component, pin].
    """
    for conn in connections:
        comp_from, pin_from = conn["from_"]
        comp_to, pin_to = conn["to"]
        if comp_from not in comp_objs or comp_to not in comp_objs:
            print(f"Connection skipped: {conn} (component not found)")
            continue
        try:
            start_pin = getattr(comp_objs[comp_from], pin_from)
            end_pin = getattr(comp_objs[comp_to], pin_to)
        except AttributeError as e:
            print(f"Error accessing pin: {e} in connection {conn}")
            continue
        d.add(elm.Wire('c').at(start_pin).to(end_pin))

def main(data):
    # Load the JSON file (adjust filename as needed)
    #with open('project.json', 'r') as f:
    #    data = json.load(f)
    
    components = data["components"]
    connections = data["connections"]
    
    # Compute positions and group information for each component.
    positions = layout_components(components)
    
    # Create the drawing and a dict to hold each component's drawn object.
    d = schemdraw.Drawing()
    d.config(fontsize=10)  # Set global font size to 10.
    comp_objs = {}
    
    for comp in components:
        name = comp["name"]
        pos, group = positions.get(name, ((0, 0), "center"))
        ic_obj = create_component(comp, pos, group)
        comp_objs[name] = ic_obj
        d.add(ic_obj)
    
    # Draw connections (wires) between pins based on the JSON.
    draw_connections(d, comp_objs, connections)
    
    # Render and save the drawing.
    d.draw()
    d.save('my_circuit.svg')
    print("Circuit drawn and saved as 'my_circuit.svg'.")

#if __name__ == "__main__":
#    main()




# what else to add?
# the type, if componenet is custom them make one, if not then just make types 